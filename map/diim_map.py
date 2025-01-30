from netCDF4 import Dataset

import numpy as np
from torch import float32,ones,tensor,empty,eye,transpose,zeros,set_num_threads
from torch import load as tload
from torch import rand as trand
from torch import argwhere as targwhere
from torch import isnan as tisnan
from torch import inverse as tinverse
from torch import transpose as ttranspose
from torch import sqrt as tsqrt
from torch import diagonal as tdiagonal
from torch.autograd.functional import jacobian as tjacobian
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import time
import sys
import os
from scipy import stats
from datetime import datetime, timedelta
from diimpy.Forward_module import Forward_Model,RRS_loss,evaluate_model_class,error_propagation
from diimpy.read_data_module import read_constants
from diimpy.CVAE_final import NN_second_layer
import pandas as pd
from mpi4py import MPI
import itertools
from numba import njit
import argparse

if 'DIIM_PATH' in os.environ:
    HOME_PATH = MODEL_HOME = os.environ["DIIM_PATH"]
else:

    print("Missing local variable DIIM_PATH. \nPlease add it with '$:export DIIM_PATH=path/to/diim'.")
    sys.exit()

    
def plot_map(ncfile=None,name_variable='sst',time_i=0,dim2 = 0,map_flag=False,map_=None):
    lats = ncfile.variables['lat'][:]
    lons = ncfile.variables['lon'][:]
    if map_flag == False:
        if len(ncfile[name_variable].shape) == 4:
            variable = ncfile[name_variable][time_i,dim2,:,:]
        else:
            variable = ncfile[name_variable][time_i,:,:]
    if map_flag != False: variable = map_
    ax = plt.axes(projection=ccrs.PlateCarree())

    plt.contourf(lons, lats, variable, 60,
             transform=ccrs.PlateCarree())

    ax.coastlines()
    
    plt.savefig('fig.pdf')
    

@njit
def linear_splines(x,xs,map_):
    
    position = 0
    for i in range(1,len(xs)-1):
        if (x<=xs[i]) and (x>xs[i-1]):
            return map_[i-1] + (map_[i] - map_[i-1])*(x - xs[i-1])/(xs[i] - xs[i-1])

@njit
def sun_declination(N):
    """
    return the declination of the sun. N is the number of days since the first of January, at 00 hours. Can be a fraction number. 
    """
    return - np.arcsin(
         0.39779 * np.cos(  (0.98565*(N + 10) + 1.914*np.sin(  0.98565*(N - 2)*np.pi/180  ))*np.pi/180 ) )#

@njit
def hour_angle(hour_,min_,sec_,lons):
    """
    return the sun hour angle as a function of the hour, min and seconds passed since the 00 UTC in that day, and the longitud.
    """
    return (hour_ + min_/60 + sec_/3600  + (lons*12/180) - 12)*np.pi/12 #local time, (lons*12/180) is degrees to hours

@njit
def zenith_angle(sun_declination_,hour_angle,lats):
    """
    return the sun zenith angle as a function of the sun declination, the hour angle, and the latitude. 
    """
    return np.arccos(
        np.sin(lats*np.pi/180)*np.sin(sun_declination_) + np.cos(lats*np.pi/180)*np.cos(sun_declination_)*np.cos(hour_angle))

def get_solar_position(time_utc,lats,lons,folder='./',dateformat='%Y%m%d-%H:%M:%S'):
    
    if os.path.exists(folder + '/zenith_'+dateformat+'.npy'):
        zenith_ = np.load(folder + '/zenith_'+datetime.strftime(time_utc + timedelta(hours=1),dateformat )+'.npy')
        return zenith_
    
    time_init = time.time()
    year = time_utc.year
    time_elapsed = time_utc  - datetime(year = year,month=1,day=1)
    hour_ = time_utc.time()
    
    days_N = time_elapsed.days + time_elapsed.seconds/86400 #86400 = 60*60*24
    hour_angle_ = hour_angle(hour_.hour,hour_.minute,hour_.second,lons)
    sun_declination_ = sun_declination(days_N)
    zenith_angle_ = zenith_angle(sun_declination_,hour_angle_,lats) 
    #print('Zenith angle computed in time:',time.time() - time_init)
    np.save(folder + '/zenith_'+datetime.strftime(time_utc + timedelta(hours=1),dateformat ) + '.npy',zenith_angle_)
    return zenith_angle_


@njit
def integrate_par(Edif,Edir,wl,irange,ilen,constant):
    PAR_ = np.zeros((Edir.shape[1:]))
    for i in range(ilen):
        PAR_ += (Edif[irange][i] + Edir[irange][i])*wl[irange][i]*1e-9 # Edif are the bin irradiance, the integral Edif for a rectangle of size dlambda

    PAR_ = PAR_  * constant
    return PAR_
    
def PAR_calculator(wl,Edir,Edif,time_utc,folder = '.',dateformat = '%Y%m%d-%H:%M:%S'):

    if os.path.exists(folder + '/PAR_'+datetime.strftime(time_utc + timedelta(hours=1),dateformat ) + '.npy'):
        PAR_ = np.load(folder + '/PAR_'+datetime.strftime(time_utc + timedelta(hours=1),dateformat ) + '.npy')
        return PAR_

    time_init = time.time()
    Na = 6.0221408e+23
    h = 6.62607015e-34
    c = 299792458 
    constant = (10**6)/ (Na*h*c)
    irange = ((wl>401) & (wl<699))
    ilen = len(wl[irange])

    PAR_ = integrate_par(Edif,Edir,wl,irange,ilen,constant)
    
    np.save(folder + '/PAR_'+datetime.strftime(time_utc + timedelta(hours=1),dateformat ) + '.npy',PAR_)
    #print('PAR computed in time:',time.time() - time_init)
    return PAR_

class read_map_class():
    def __init__(self,oasim_data_file = '/path/where/to/read/oasim/data',rrs_data_file='/path/where/to/read/rrs/data',\
                 time_utc = datetime(year=2020,month=3,day=1),PAR_folder = '/path/where/to/read/PAR',\
                 zenith_folder = '/path/where/to/read/zenith/angle',my_precision = float32,\
                 scratch_path = '/path/to/scratch/',store_data = False,dateformat = '%Y%m%d %H:%M:%S'):
        
        init_time = time.time()
        self.my_precision = my_precision
        self.date_str = datetime.strftime(time_utc + timedelta(hours=1),dateformat ) 

        def read_general_parameters(self,parameters_csv):
            general_parameters_ = pd.read_csv(parameters_csv)
            self.len_lat = int(general_parameters_['len_lat'].iloc[0])
            self.len_lon = int(general_parameters_['len_lon'].iloc[0])
            self.n_points = int(general_parameters_['n_points'].iloc[0])
            self.mesh_lat_path = str(general_parameters_['mesh_lat_path'].iloc[0])
            self.mesh_lon_path = str(general_parameters_['mesh_lon_path'].iloc[0])
            self.input_data_path = str(general_parameters_['input_data_path'].iloc[0])
            self.non_nan_points_path = str(general_parameters_['non_nan_points_path'].iloc[0])
            self.non_nan_points_len = int(general_parameters_['non_nan_points_len'].iloc[0])
            self.time = str(general_parameters_['time'].iloc[0])
            self.time_format = str(general_parameters_['time_format'].iloc[0])

        if store_data == True:
            general_parameters = pd.DataFrame()
            
            OASIM_data = Dataset(oasim_data_file)
            RRS_data = Dataset(rrs_data_file)
            
            wl = OASIM_data['wavelength'][:].data
            
            general_parameters['len_lat']  = [len(OASIM_data['lat'][:])]
            general_parameters['len_lon']  = [len(OASIM_data['lon'][:])]
            
            general_parameters['n_points']  = [len(OASIM_data['lat'][:])*len(OASIM_data['lon'][:])]
            n_points = int(general_parameters['n_points'].iloc[0])
            
            mesh_lat,mesh_lon = np.meshgrid(OASIM_data['lat'][:],OASIM_data['lon'][:])
            mesh_lat, mesh_lon = mesh_lat.T.data, mesh_lon.T.data
            general_parameters['mesh_lat_path'] = [scratch_path + '/mesh_lat.npy']
            np.save(scratch_path + '/mesh_lat.npy',mesh_lat)
            general_parameters['mesh_lon_path'] = [scratch_path + '/mesh_lon.npy']
            np.save(scratch_path + '/mesh_lon.npy',mesh_lon)
        
            x_data = np.empty((n_points,17))
            
            Edif = np.ma.filled(np.mean(OASIM_data['esout'][:,:,:,:], axis = 0),fill_value=-999)
            x_data[:,0] = linear_splines(412.5,wl,Edif).reshape((n_points))
            x_data[:,1] = linear_splines(442.5,wl,Edif).reshape((n_points))
            x_data[:,2] = linear_splines(490,wl,Edif).reshape((n_points))
            x_data[:,3] = linear_splines(510,wl,Edif).reshape((n_points))
            x_data[:,4] = linear_splines(555,wl,Edif).reshape((n_points))
            
            Edir = np.ma.filled(np.mean(OASIM_data['edout'][:,:,:,:], axis = 0),fill_value=-999)
            x_data[:,5] = linear_splines(412.5,wl,Edir).reshape((n_points))
            x_data[:,6] = linear_splines(442.5,wl,Edir).reshape((n_points))
            x_data[:,7] = linear_splines(490,wl,Edir).reshape((n_points))
            x_data[:,8] = linear_splines(510,wl,Edir).reshape((n_points))
            x_data[:,9] = linear_splines(555,wl,Edir).reshape((n_points))
            
            x_data[:,10] = ones((n_points)) * 412.5
            x_data[:,11] = ones((n_points)) * 442.5
            x_data[:,12] = ones((n_points)) * 490
            x_data[:,13] = ones((n_points)) * 510
            x_data[:,14] = ones((n_points)) * 555
            general_parameters['time'] = [time_utc]
            general_parameters['time_format'] = [dateformat]
            x_data[:,15] = get_solar_position(time_utc,mesh_lat,mesh_lon,folder = zenith_folder,dateformat = dateformat).reshape((n_points))
            
            x_data[:,16] = PAR_calculator(wl,Edir,Edif,time_utc,folder = PAR_folder,dateformat = dateformat).reshape((n_points))
            
            y_data = np.empty((n_points,5))
            y_data[:,0] = RRS_data['RRS412'][0,:,:].reshape((n_points))
            y_data[:,1] = RRS_data['RRS443'][0,:,:].reshape((n_points))
            y_data[:,2] = RRS_data['RRS490'][0,:,:].reshape((n_points))
            y_data[:,3] = RRS_data['RRS510'][0,:,:].reshape((n_points))
            y_data[:,4] = RRS_data['RRS555'][0,:,:].reshape((n_points))

            nan_y_points = np.any(y_data<-900,axis=1)
            nan_x_points = np.any(x_data<-900,axis=1)
            nan_points_ = nan_y_points | nan_x_points
            nan_points = np.where(nan_points_ == True)[0]
            non_nan_points = np.where(nan_points_ == False)[0]

            general_parameters['input_data_path'] = [scratch_path + '/input_data']
            if not os.path.exists(scratch_path + '/input_data'): os.mkdir(scratch_path + '/input_data')
            np.save(scratch_path + '/input_data/x_data_'+self.date_str+'.npy',x_data[non_nan_points])
            np.save(scratch_path + '/input_data/y_data_'+self.date_str+'.npy',y_data[non_nan_points])
            
            general_parameters['non_nan_points_path'] = [scratch_path + '/non_nan_points_'+self.date_str+'.npy']
            np.save(scratch_path + '/non_nan_points_'+self.date_str+'.npy',non_nan_points)
            general_parameters['non_nan_points_len'] = [len(non_nan_points)]

            
            general_parameters.to_csv(scratch_path + '/general_parameters_'+self.date_str+'.csv')
            
            y_data[y_data < -900.] = np.nan
            x_data[x_data < -900.] = np.nan
            self.x_data = tensor(x_data[non_nan_points]).to(my_precision)
            self.y_data = tensor(y_data[non_nan_points]).to(my_precision)
            self.non_nan_points = non_nan_points
            read_general_parameters(self,scratch_path + '/general_parameters_'+self.date_str+'.csv')
            
        elif os.path.exists(scratch_path + '/general_parameters_'+self.date_str+'.csv'):
            
            read_general_parameters(self,scratch_path + '/general_parameters_'+self.date_str+'.csv')
            y_data = np.load(self.input_data_path + '/y_data_'+self.date_str+'.npy')
            x_data = np.load(self.input_data_path + '/x_data_'+self.date_str+'.npy')
            y_data[y_data < -900.] = np.nan
            x_data[x_data < -900.] = np.nan
            self.x_data = tensor(x_data).to(my_precision)
            self.y_data = tensor(y_data).to(my_precision)
            self.non_nan_points = tensor(np.load(self.non_nan_points_path))
            
        else:
            print(scratch_path + "/general_parameters_"+self.date_str+".csv dosn't exist,")
            print("please enter the scratch_path where the input data and general_parameters_"+dateformat+".csv was stores,")
            print("or use the parameter 'store_data = True'")
        #print('data loaded..., time: ',time.time() - init_time )
        

    def __len__(self):
        return self.non_nan_points_len

    
    def __getitem__(self, idx):
        image = empty((5,5))
        image[:,0] = self.x_data[idx][:5]
        image[:,1] = self.x_data[idx][5:10]
        image[:,2] = self.x_data[idx][10:15]
        image[:,3] = self.x_data[idx][15]
        image[:,4] = self.x_data[idx][16]

        label = self.y_data[idx]

        return image,label
    
    def get_coordinates(self):
        mesh_lat= np.load(self.mesh_lat_path)
        mesh_lon = np.load(self.mesh_lon_path)
        return mesh_lat,mesh_lon

    

def local_initial_conditions_nn(F_model,constant,data,precision = float32,my_device = 'cpu',rearange_needed=True):
    
    init_time_ = time.time()

    model_NN = NN_second_layer(my_device=my_device,chla_centered=True ,precision = precision,constant=constant).to(my_device)

    model_NN.load_state_dict(tload(MODEL_HOME + '/settings/VAE_model/model_second_part_chla_centered.pt'))
    model_NN.eval()

    if rearange_needed == True:
        X,Y = data
        data_rearanged = empty((X.shape[0],1,17))
        data_rearanged[:,0,:5] = Y
        data_rearanged[:,0,5:10] = X[:,0]
        data_rearanged[:,0,10:15] = X[:,1]
        data_rearanged[:,0,15] = X[:,3,0]
        data_rearanged[:,0,16] = X[:,4,0]
    else:
        data_rearanged = data
        
    z_hat,cov_z,mu_z,kd_hat,bbp_hat,rrs_hat = model_NN(data_rearanged) #we are working with \lambda as imput, but the NN dosent use it.
    mu_z = (mu_z* model_NN.y_mul[0] + model_NN.y_add[0]).clone().detach()

    if tisnan(mu_z).any():
        mu_z[targwhere(tisnan(mu_z))] = trand(mu_z[targwhere(tisnan(mu_z))].shape,dtype=float32)

    state_dict = F_model.state_dict()
    state_dict['chparam']  = mu_z.unsqueeze(1)
    F_model.load_state_dict(state_dict)
    #print('time for NN data,', time.time() - init_time_)
    return mu_z

def get_jacobian_components(original_jacobian,len_: int,comp: int):
    new_jacobian = empty((len_,comp,3))
    map(lambda i: new_jacobian[i].append(original_jacobian[i,:,i,:,:].reshape(comp,3)),range(len_))
    return new_jacobian

def train_loop(data_i,model,loss_fn,optimizer,N,kind='all',num_days=1,my_device = 'cpu',constant = None,perturbation_factors_ = None,calc_kd=True,calc_bbp=True):
    """
    The train loop evaluates the Remote Sensing Reflectance RRS for each wavelength >>>pred=model(data_i), evaluates the loss function
    >>>loss=loss_fn(pred,y), evaluates the gradient of RRS with respect to the parameters, >>>loss.backward(), modifies the value of the parameters according to the optimizer criterium, >>>optimizer.step(),
    sets the gradient of RRS to zero. After this, compute the approximate covariance matrix of the active constituents to, finally, compute kd and bbp with uncertainty. 
    
    Parameters:

      data_i: tuple of datasets (X,Y), X is the input data E_dir(lambda), E_dif(lambda), lambda, zenith(lambda) and PAR(lambda), a tensor of dimension (*,5,5), where * stands for the number of days evaluated. Y is the satellite data RRS(lambda), a tensor of dimension (*,5).
     
      model: RRS. Is a pytorch object, with chparam as learnable parameters of dimension $(*,1,3)$, a variable called perturbation_factors_\
             of dimension 14, and a forward function that returns the predicted RRS for the five values of lambda 412.5, 442.5, 490, 510 and 555. 
      
      loss_fn: The RRS_loss object, is an object with the variables 
               x_a: mean value of the prior values for chla, nap and cdom, with dimension (3).
               s_a: covariance matrix of the prior for chla, nap and cdom, dimension (3,3).
               s_e: covariance matrix of RRS. Dimension (5,5).
               s_a_inverse: the inverse of s_a
               s_e_inverse: the inverse of s_e

      optimizer: a pytorch optimizer object. 

      N: the maximum number of iterations for the train loop. 

      kind: can be "all", "parameter_estimation" or "rrs". 
            if kind = "all", train_loop returns a dictionary with {'X_hat':X_hat,'kd_hat':kd_hat,'bbp_hat':bbp_hat,'RRS_hat':last_rrs}, where X_hat is the predicted\
            chlorophyll, NAP and CDOM, kd_hat is the predicted down light attenuation coefficient, bbp_hat is the predicted backward scattering and RRS is \
            the predicted Remote Sensing Reflectance. All with uncertainty.  Example: X_hat[0] = [X_hat_412.5[0],X_hat_412_delta[0],X_hat_442[0],X_hat_442_delta[0],...,X_hat_555_delta[0]].
            If kind = "parameter_estimation", train_loop returns a (*,9) tensor, with the first element being the prediction of chlorophyll,\
            the next 5 the predictions for kd, and the last three, the predictions for bbp. 
            If kind = "rrs", train_loop will return the prediction of RRS, the history of predictions for chlorophyll, NAP and CDOM, and the history of loss_functions, for all the iterations on the loop. 

      num_days: The number of days for which the inversion is being performed. Usually is the first dimension of X or Y in data_i. Default is equal to 1. 
      
      my_device: torch devise, could be 'cpu', 'cuda', 'mps', etc. 
      
      constant: Set of constants for the model. Is a dictionary, necessary for running the forward model. See diimpy/read_data_module.py.

      perturbation factors_: torch tensor of dimension (14). Numbers that multiply the constants of the forward model. 

      Scheduler: Defines if use a scheduler in the Adam Algorithm, it can accelerate the convergence of the algorithm. 
    """
    past_pred=empty((N,num_days,3))

    criterium = 1
    criterium_2 = 0
    i=0
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    X = data_i[0]
    Y = data_i[1]
    s_a = (loss_fn.s_a)
    s_e = (loss_fn.s_e)
    s_e_inverse = (loss_fn.s_e_inverse)
    s_a_inverse = (loss_fn.s_a_inverse)

    dR = (1e-13)
    ls_val=[]
    init_time = time.time()
    while (((criterium >dR ) & (i<N)) or ((criterium_2 < 100)&(i<N))):
        
        pred = model(X,constant = constant,perturbation_factors_ = perturbation_factors_)
        
        if tisnan(pred).any():
            where_nan = targwhere(tisnan(pred))[0]
            if where_nan[0] == 0:
                pred[where_nan] = (pred[where_nan[0]+1,where_nan[1]].clone())
            elif where_nan[0] == len(pred[0]):
                pred[where_nan] = (pred[where_nan[0]-1,where_nan[1]].clone())
            else:
                pred[where_nan] = (pred[where_nan[0]+1,where_nan[1]].clone() + pred[where_nan[0]-1,where_nan[1]].clone())/2
                
        loss = loss_fn(Y,pred,model.state_dict()['chparam'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        ls_val.append(loss.item())
        past_pred[i] = model.state_dict()['chparam'][:,0,:]

        if i != 0:
            criterium = ls_val[-2] - ls_val[-1]
        if criterium <=dR:
            criterium_2+=1
        i+=1
        scheduler.step(loss)
        
    last_rrs = pred.clone().detach()
    last_i = i
    print('one loop',time.time() - init_time)
    init_time = time.time()
    
    parameters_eval = list(model.parameters())[0].clone().detach()
    evaluate_model = evaluate_model_class(model=model,X=X,constant = constant)
    

    K_x_ = tjacobian(evaluate_model.model_der,inputs=(parameters_eval))
    K_x = get_jacobian_components(K_x_,len(parameters_eval),5)
    del K_x_
        
    S_hat = tinverse( ttranspose(K_x,1,2) @ ( s_e_inverse @ K_x ) + s_a_inverse  )
        
    X_hat = np.empty((len(parameters_eval),6))
    X_hat[:,::2] = past_pred[last_i-1].clone().detach()
    X_hat[:,1::2] = tsqrt(tdiagonal(S_hat,dim1=1,dim2=2).clone().detach())
    output = {'X_hat':X_hat,'RRS_hat':last_rrs}
    print('uncertainty computation',time.time() - init_time)
    if calc_kd:
        init_time = time.time()

        kd_values = evaluate_model.kd_der(parameters_eval)
        kd_hat = empty((len(parameters_eval),10))
        kd_derivative_ = tjacobian(evaluate_model.kd_der,inputs=(parameters_eval))
        kd_derivative = get_jacobian_components(kd_derivative_,len(parameters_eval),5)
        del kd_derivative_

        kd_delta = error_propagation(kd_derivative,S_hat)
        kd_hat[:,::2] = kd_values.clone().detach()
        kd_hat[:,1::2] = tsqrt(kd_delta).clone().detach()
        output['kd_hat'] = kd_hat
    if calc_bbp:

        bbp_values = evaluate_model.bbp_der(parameters_eval)
        bbp_hat = empty((len(parameters_eval),6))
        bbp_values = evaluate_model.bbp_der(parameters_eval)
    
        bbp_derivative_ = tjacobian(evaluate_model.bbp_der,inputs=(parameters_eval))
        bbp_derivative = get_jacobian_components(bbp_derivative_,len(parameters_eval),3)
        del bbp_derivative_
        
        bbp_delta = error_propagation(bbp_derivative,S_hat)
        bbp_hat[:,::2] = bbp_values.clone().detach()
        bbp_hat[:,1::2] = tsqrt(bbp_delta).clone().detach()
        output['bbp_hat'] = bbp_hat
    
    print('kd and bbp',time.time() - init_time)
    sys.exit()
    
    #print("time for training...",time.time() - time_init)
    return output


def create_nan_array(shape):
        nan_array = np.empty(shape)
        nan_array[:] = np.nan
        return nan_array

def arguments():
    parser = argparse.ArgumentParser(description = '''
    Script to create a map with the output of DIIM.
    ''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument('--conf_file', action='store_true', help="Read the configuration values from file. ")
    parser.add_argument('--conf_file_name','-iconf',
                        type = str,
                        required = False,
                        help = '''Name of the conf_file, used to read the configurations values if --conf_file is used.''',
                        default = 'conf_file.csv'
                        )
    
    parser.add_argument('--dateformat', '-df',
                        type = str,
                        required = False,
                        help = '''Date format for reading and writing file names. e.g. prefix.%%Y%%m%%d-%%H:%%M:%%S.nc''',
                        default = '%Y%m%d-%H:%M:%S'
                        )
    parser.add_argument('--scratch_path', '-os',
                        type = str,
                        required = False,
                        help = '''Path to store scratch files, including metadata to create the map.''',
                        default = HOME_PATH + '/scratch'
                        )
    parser.add_argument('--perturbation_factors_path', '-ip',
                        type = str,
                        required = False,
                        help = '''Path where to read the value for the perturbation factors to use.''',
                        default = HOME_PATH + '/settings/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy'
                        )
    parser.add_argument('--device', '-d',
                        type = str,
                        required = False,
                        help = '''device to be used for torch objects. (cpu or cuda). Advaced cpu, since cuda has not being tested.''',
                        default = 'cpu'
                        )
    parser.add_argument('--constants_path', '-ic',
                        type = str,
                        required = False,
                        help = '''Path where to read the value for the constants of the forward model.''',
                        default = HOME_PATH + '/settings'
                        )
    parser.add_argument('--oasim_data_path', '-io',
                        type = str,
                        required = False,
                        help = '''Path where to read the imputs from the OASIM model.''',
                        default = HOME_PATH + '/map/oasim_map/maps/OASIM_MED.nc'
                        )
    parser.add_argument('--rrs_data_path', '-ir',
                        type = str,
                        required = False,
                        help = '''Path where to read the imputs from Remote Sensing Reflectance.''',
                        default = HOME_PATH + '/map/RRS/RRS_MED.nc'
                        )
    parser.add_argument('--PAR_path', '-ipar',
                        type = str,
                        required = False,
                        help = '''Path where to write and read the values for PAR.''',
                        default = HOME_PATH + '/map/PAR'
                        )
    parser.add_argument('--zenith_path', '-izen',
                        type = str,
                        required = False,
                        help = '''Path where to write and read the values for zenith angle.''',
                        default = HOME_PATH + '/map/zenith'
                        )
    parser.add_argument('--date', '-t',
                        type = str,
                        required = False,
                        help = '''Date of the map to be created .''',
                        default = '20000101-00:00:00'
                        )
    parser.add_argument('--output_path', '-o',
                        type = str,
                        required = False,
                        help = '''Path where to write the output of the inversion process.''',
                        default = HOME_PATH + '/map/diim_maps'
                        )
    return parser


def read_parameters():
    parser = arguments()
    args = parser.parse_args()
    default_values = {action.dest: action.default for action in parser._actions if action.default is not argparse.SUPPRESS}
    if args.conf_file:
        conf_ = pd.read_csv(args.conf_file_name,sep = ' ',index_col = 0,comment = '#').T.reset_index().rename(columns={'index':'scratch_path'})
        conf_ = {name_:conf_[name_][0] for name_ in conf_.keys()}

    else:
        conf_ = default_values

    for action in parser._actions:
        if any([(option in sys.argv) for option in action.option_strings]):
            conf_[action.dest] = vars(args)[action.dest]
            
    return conf_

def set_mpi(master_addr,master_port,num_threads):
    comm = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nranks = comm.size
    #in case the code try to be run in GPU, lets specify master as local host. 
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    set_num_threads(num_threads)
    return comm,rank,nranks

def inversion(dateformat='%Y%m%d-%H:%M:%S',scratch_path=HOME_PATH + '/scratch',oasim_data_path=HOME_PATH + '/map/oasim_map/maps/OASIM_MED.nc',rrs_data_path=HOME_PATH + '/map/RRS/RRS_MED.nc',PAR_path=HOME_PATH + '/map/PAR',zenith_path=HOME_PATH + '/map/zenith',date_str='20000101-00:00:00',rank=0,nranks=1,comm=None,my_device='cpu',perturbation_factors=ones(14, dtype=float32),constant=None,calc_kd=True,calc_bbp=True):
    
    """
    Inversts the rrs, and store the outputs in scratch_path. These outputs are intented to be used to create a netcdf with an independent function, later. Use in combination of mpi4py, comm is used to place barriers. 
    
    Parameters:

        dateformat
            Date format for reading and writing file names. e.g. prefix.%%Y%%m%%d-%%H:%%M:%%S.nc. default: '%Y%m%d-%H:%M:%S'.

        scratch_path
            Path to store scratch files, including metadata to create the map. default: HOME_PATH + '/scratch'

        oasim_data_path
            Path where to read the imputs from the OASIM model. default: HOME_PATH + '/map/oasim_map/maps/OASIM_MED.nc'

        rrs_data_path
            Path where to read the imputs from Remote Sensing Reflectance. default: HOME_PATH + '/map/RRS/RRS_MED.nc'

        PAR_path
            Path where to write and read the values for PAR. default: HOME_PATH + '/map/PAR'

        zenith_path
            Path where to write and read the values for zenith angle.

        date_str
            Date of the map to be created. default: '20000101-00:00:00'

        rank
            comm.Get_rank()
        nranks
            comm.size
        comm
            comm.Get_rank()
    """
    
    date = datetime.strptime(date_str,dateformat)
    time_utc = datetime(year = date.year,month = date.month,day=date.day,hour=11)

    if rank == 0 :
        dataset = read_map_class(oasim_data_file=oasim_data_path,\
                                 rrs_data_file=rrs_data_path,time_utc = time_utc,scratch_path = scratch_path ,\
                                 PAR_folder = PAR_path,zenith_folder = zenith_path,store_data = False,dateformat = dateformat)
    comm.Barrier()

    
    if rank != 0 :
        dataset = read_map_class(oasim_data_file=oasim_data_path,\
                                 rrs_data_file=rrs_data_path,time_utc = time_utc,scratch_path = scratch_path ,\
                                 PAR_folder = PAR_path,zenith_folder = zenith_path,store_data = False,dateformat = dateformat)
            
    lr = 0.029853826189179603
    x_a = zeros(3)
    s_a = eye(3) * 4.9
    s_e = (eye(5)*tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf

    batch_size = 10#len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if rank == 0:
        chla_NN_rec = np.zeros((len(dataset),3), dtype=np.float32)
        chla_log_rec = np.zeros((len(dataset),6), dtype=np.float32)
        kd_log_rec = np.zeros((len(dataset),10), dtype=np.float32)
        bbp_log_rec = np.zeros((len(dataset),6), dtype=np.float32)
        RRS_log_rec = np.zeros((len(dataset),5), dtype=np.float32)
        
    if nranks == 1:
        n_workers = lenght_iter = 1
        worker_i = rank 
    elif rank!=0:
        n_workers = lenght_iter = nranks - 1 #he iterates every lenght_iter = n_workers elements.
        worker_i = rank - 1
    else:
        n_workers = nranks - 1
        lenght_iter = 1 #he iterates for all elements, because needs to receive all of them. 
        worker_i = 0

    #if rank == 0:
    #print('starting to process the map with {} points, by grouping them in groups of {}, and working with {} simultaneus processes'.format(len(dataset.x_data),batch_size,n_workers))
    comm.Barrier()
    
    for i,data in itertools.islice(enumerate(dataloader),worker_i,None,lenght_iter):
        
        if (nranks == 1) or (rank != 0):
            len_batch = len(data[0])
            
            model = Forward_Model(num_days=len_batch).to(my_device)
            model.perturbation_factors = perturbation_factors
            chla_NN_sen = local_initial_conditions_nn(model,constant,data,precision = float32,my_device = 'cpu').numpy()
            loss = RRS_loss(x_a,s_a,s_e,num_days=len_batch,my_device = my_device)
            optimizer = Adam(model.parameters(),lr=lr)
            output = train_loop(data,model,loss,optimizer,4000,kind='all',\
                                      num_days=len_batch,constant = constant,perturbation_factors_ = perturbation_factors)

            if np.isnan(output['X_hat']).all():
                model = Forward_Model(num_days=len_batch).to(my_device)
                model.perturbation_factors = perturbation_factors
                loss = RRS_loss(x_a,s_a,s_e,num_days=len_batch,my_device = my_device)
                optimizer = Adam(model.parameters(),lr=lr)
                output = train_loop(data,model,loss,optimizer,4000,kind='all',\
                                      num_days=len_batch,constant = constant,perturbation_factors_ = perturbation_factors,calc_kd = calc_kd, calc_bbp = calc_bbp)
                if np.isnan(output['X_hat']).all():
                    print(rank,i,'had a nan value for date, ',date_str)
            
            chla_log_sen = output['X_hat']
            RRS_log_sen = output['RRS_hat']
            if calc_kd: kd_log_sen = output['kd_hat']
            if calc_bbp: bbp_log_sen = output['bbp_hat']
            
            


            if nranks!=1:
                comm.Send([chla_NN_sen.astype(np.float32), MPI.FLOAT],dest=0,tag=1)
                comm.Send([chla_log_sen.astype(np.float32), MPI.FLOAT],dest=0,tag=2)
                comm.Send([RRS_log_sen.numpy().astype(np.float32), MPI.FLOAT],dest=0,tag=5)
                if calc_kd: comm.Send([kd_log_sen.numpy().astype(np.float32), MPI.FLOAT],dest=0,tag=3)
                if calc_bbp: comm.Send([bbp_log_sen.numpy().astype(np.float32), MPI.FLOAT],dest=0,tag=4)

                del chla_NN_sen
                del chla_log_sen
                del RRS_log_sen
                if calc_kd: del kd_log_sen
                if calc_bbp: del bbp_log_sen
                del model
                del output
                del optimizer
                del loss

            else:
                chla_NN_rec[batch_size*i:batch_size*i + len_batch] = chla_NN_sen
                chla_log_rec[batch_size*i:batch_size*i + len_batch] = chla_log_sen
                RRS_log_rec[batch_size*i:batch_size*i + len_batch] = RRS_log_sen
                if calc_kd: kd_log_rec[batch_size*i:batch_size*i + len_batch] = kd_log_sen
                if calc_bbp: bbp_log_rec[batch_size*i:batch_size*i + len_batch] = bbp_log_sen
                
                del chla_NN_sen
                del chla_log_sen
                del RRS_log_sen
                if calc_kd: del kd_log_sen
                if calc_bbp: del bbp_log_sen
                del model
                del output
                del optimizer
                del loss

        elif (rank == 0) & (nranks!=1):
            len_batch = len(data[0])
            rank_of_worker_sender = (i%n_workers) + 1
            comm.Recv([ chla_NN_rec[batch_size*i:batch_size*i + len_batch],MPI.FLOAT],source=rank_of_worker_sender,tag=1 )
            comm.Recv([ chla_log_rec[batch_size*i:batch_size*i + len_batch],MPI.FLOAT],source=rank_of_worker_sender,tag=2 )
            comm.Recv([ RRS_log_rec[batch_size*i:batch_size*i + len_batch],MPI.FLOAT],source=rank_of_worker_sender,tag=5 )
            if calc_kd: comm.Recv([ kd_log_rec[batch_size*i:batch_size*i + len_batch],MPI.FLOAT],source=rank_of_worker_sender,tag=3 )
            if calc_bbp: comm.Recv([ bbp_log_rec[batch_size*i:batch_size*i + len_batch],MPI.FLOAT],source=rank_of_worker_sender,tag=4 )

            #print('communication received from rank {}, corresponding to elements from {} to {}'.format(rank_of_worker_sender,batch_size*i,batch_size*i + len_batch))
            
    if rank == 0:
        np.save(scratch_path+'/chla_NN_'+date_str+'.npy',chla_NN_rec)
        np.save(scratch_path+'/chla_log_'+date_str+'.npy',chla_log_rec)
        np.save(scratch_path+'/RRS_log_'+date_str+'.npy',RRS_log_rec)
        if calc_kd: np.save(scratch_path+'/kd_log_'+date_str+'.npy',kd_log_rec)
        if calc_bbp: np.save(scratch_path+'/bbp_log_'+date_str+'.npy',bbp_log_rec)
        
    comm.Barrier()
    return dataset

def create_map(scratch_path,dataset,output_path='./map.nc',date_str = '20000101',calc_kd=True,calc_bbp=True,calc_NN = True,calc_rrs = True,calc_chla = True):
    if calc_NN: chla_NN_d = np.load(scratch_path + '/chla_NN_'+date_str+'.npy')
    if calc_NN: chla_NN_d[np.isinf(chla_NN_d)] = np.nan
    if calc_chla: chla_log_d = np.load(scratch_path + '/chla_log_'+date_str+'.npy')
    if calc_chla: chla_log_d[np.isinf(chla_log_d)] = np.nan
    if calc_kd: kd_d = np.load(scratch_path + '/kd_log_'+date_str+'.npy')
    if calc_kd: kd_d[np.isinf(kd_d)] = np.nan
    if calc_bbp: bbp_d = np.load(scratch_path + '/bbp_log_'+date_str+'.npy')
    if calc_bbp: bbp_d[np.isinf(bbp_d)] = np.nan
    if calc_rrs: RRS_d = np.load(scratch_path + '/RRS_log_'+date_str+'.npy')
    if calc_rrs: RRS_d[np.isinf(RRS_d)]
    
    len_lat = dataset.len_lat
    len_lon = dataset.len_lon

    no_nan_points = np.load(dataset.non_nan_points_path)

    def init_map(lambda_dependent=True,no_nan_values = -999.0,is_bbp=False):
        if lambda_dependent == True:
            map_ = np.zeros((5,len_lat*len_lon))*np.nan
            if is_bbp == False:
                map_[:,no_nan_points] = no_nan_values
            else:
                map_[1,no_nan_points] = no_nan_values[0,:]
                map_[2,no_nan_points] = no_nan_values[0,:]
                map_[4,no_nan_points] = no_nan_values[0,:]
            map_ = np.reshape(map_,(map_.shape[0],len_lat,len_lon))
            

        else:
            map_ = np.zeros((len_lat*len_lon)) * np.nan
            map_[no_nan_points] = no_nan_values
            map_ = np.reshape(map_,(len_lat,len_lon))

        return map_

    #test = np.zeros(chla_log_d[:,0].shape)
    #test[np.argwhere(chla_log_d[:,0]==-2.008570432662964)] = 1
    #test_ = np.zeros(len_lat*len_lon)
    #test_[no_nan_points] = test
    #test_ = np.reshape(test_,(len_lat,len_lon))
    #print(np.argwhere(test==1)[0])
    #print(np.argwhere(test_==1)[0])
    #sys.exit() 
    
    if calc_NN:
        chla_NN = init_map(lambda_dependent = False,no_nan_values = chla_NN_d[:len(dataset),0])
        nap_NN = init_map(lambda_dependent = False,no_nan_values = chla_NN_d[:len(dataset),1])
        cdom_NN = init_map(lambda_dependent = False,no_nan_values = chla_NN_d[:len(dataset),2])
        
    
    if calc_chla:
        chla = init_map(lambda_dependent = False,no_nan_values = chla_log_d[:len(dataset),0])
        dchla = init_map(lambda_dependent = False,no_nan_values = chla_log_d[:len(dataset),1])
        nap = init_map(lambda_dependent = False,no_nan_values = chla_log_d[:len(dataset),2])
        dnap = init_map(lambda_dependent = False,no_nan_values = chla_log_d[:len(dataset),3])
        cdom = init_map(lambda_dependent = False,no_nan_values = chla_log_d[:len(dataset),4])
        dcdom = init_map(lambda_dependent = False,no_nan_values = chla_log_d[:len(dataset),5])
    
        del chla_NN_d
        del chla_log_d

    if calc_kd:
        kd = init_map(no_nan_values = kd_d[:len(dataset),::2].T)
        dkd = init_map(no_nan_values = kd_d[:len(dataset),1::2].T)
        del kd_d

    if calc_bbp:
        bbp = init_map(no_nan_values = bbp_d[:len(dataset),::2].T,is_bbp=True)
        dbbp = init_map(no_nan_values = bbp_d[:len(dataset),1::2].T,is_bbp=True)
        del bbp_d

    if calc_rrs:
        RRS = init_map(no_nan_values = RRS_d[:len(dataset)].T)
        del RRS_d
    

    mesh_lat = np.load(dataset.mesh_lat_path)
    mesh_lon = np.load(dataset.mesh_lon_path)

    lon,lat = mesh_lon[0],mesh_lat[:,0]

    ncfile = group =  Dataset(output_path,mode='w',format='NETCDF4_CLASSIC')
    lat_dim = ncfile.createDimension('lat', len(lat))     # latitude axis
    lon_dim = ncfile.createDimension('lon', len(lon))    # longitude axis
    time_dim = ncfile.createDimension('time', 1) # unlimited axis (can be appended to).
    lambda_dim = ncfile.createDimension('lambda',5)

    group.parameter_code = 'Inversion'
    group.parameters_description = 'chlorophyll, Non algal particles, Colored dissolved organic matter, Remote Sensing Reflectance, Backward scattering coefficient and light atenuation coefficient from the inversion model https://doi.org/10.5194/gmd-2024-174'
    group.data_date = date_str
    group.creation_date = datetime.utcnow().strftime('%a %b %d %Y')
    group.creation_time = datetime.utcnow().strftime('%H:%M:%S UTC')
    group.westernmost_longitude = np.min(lon)
    group.easternmost_longitude = np.max(lon)
    group.southernmost_latitude = np.min(lat)
    group.northernmost_latitude = np.max(lat)
    group.grid_resolution =  '~ 1 Km'
    group.contact = 'carlos.soto362@gmail.com'
    group.netcdf_version = 'v4'
    group.grid_mapping = 'Equirectangular'
    group.software_name = 'Data-Informed Inversion Model (DIIM)'
    group.source = 'surface inversion'
    group.citation = ' Soto López, C. E., Anselmi, F., Gharbi Dit Kacem, M., and Lazzari, P.: Data-Informed Inversion Model (DIIM): a framework to retrieve marine optical constituents in the BOUSSOLE site using a three-stream irradiance model, Geosci. Model Dev. Discuss. [preprint], https://doi.org/10.5194/gmd-2024-174, in review, 2024'
    group.institution = 'OGS'
    group.software_version = 'v1.0'
    group.site_name = 'MED'
    group.product_version = 'v01'

    
    time_var = group.createVariable('time','f4',('time',))
    lat_var = group.createVariable('lat','f4',('lat',))
    lon_var = group.createVariable('lon','f4',('lon',))
    lambda_var = group.createVariable('lambda','f4',('lambda'))

    if calc_NN:
        chla_NN_var = group.createVariable('chla_NN','f4',('time','lat','lon'))
        nap_NN_var = group.createVariable('nap_NN','f4',('time','lat','lon'))
        cdom_NN_var = group.createVariable('cdom_NN','f4',('time','lat','lon'))
    if calc_chla:
        chla_var = group.createVariable('chla','f4',('time','lat','lon'))
        nap_var = group.createVariable('nap','f4',('time','lat','lon'))
        cdom_var = group.createVariable('cdom','f4',('time','lat','lon'))
        dchla_var = group.createVariable('dchla','f4',('time','lat','lon'))
        dnap_var = group.createVariable('dnap','f4',('time','lat','lon'))
        dcdom_var = group.createVariable('dcdom','f4',('time','lat','lon'))
    if calc_kd:
        kd_var = group.createVariable('kd','f4',('time','lambda','lat','lon'))
        dkd_var = group.createVariable('dkd','f4',('time','lambda','lat','lon'))
    if calc_bbp:
        bbp_var = group.createVariable('bbp','f4',('time','lambda','lat','lon'))
        dbbp_var = group.createVariable('dbbp','f4',('time','lambda','lat','lon'))
    if calc_rrs: RRS_var = group.createVariable('RRS','f4',('time','lambda','lat','lon'))    
    
    def fill_atributes(var,long_name,axis,units,calendar='Gregorian',valid_min=False,valid_max=False,missing_value=-999.0):
        var.long_name = long_name
        if axis: var.axis = axis
        var.units = units
        if calendar: var.calendar = calendar
        if valid_max: var.valid_max = valid_max
        if valid_min: var.valid_min = valid_min
        if missing_value: var.missing_value = missing_value
        
    fill_atributes(time_var,long_name='Reference time',axis='T',units='seconds since 1981-01-01 00:00:00',valid_max=False,valid_min=False,missing_value=False)
    fill_atributes(lat_var,long_name='latitute',axis='Y',units='degrees_north',valid_min=np.min(lat),valid_max=np.max(lat),missing_value=False)
    fill_atributes(lon_var,long_name = 'longitude',axis='X',units='degrees_east',valid_min=np.min(lon),valid_max=np.max(lon),missing_value=False)
    fill_atributes(lon_var,long_name = 'longitude',axis='X',units='degrees_east',valid_min=np.min(lon),valid_max=np.max(lon),missing_value=False)
    fill_atributes(lambda_var,long_name='wavelenght',axis='lambda',units='nanometers',valid_min = 412.5,valid_max=555,missing_value=False)
    
    if calc_NN:
        fill_atributes(chla_NN_var,long_name = 'Neural Network estimate of the logarithm of the concentration of Chlorophyll', axis=False,units='log [mg m^(-3)]',valid_max = np.nanmax(chla_NN),valid_min = np.nanmin(chla_NN[chla_NN>-990]))
        fill_atributes(nap_NN_var,long_name = 'Neural Network estimate of the logarithm of the concentration of Non Algal Particles', axis=False,units='log [mg m^(-3)]',valid_max = np.nanmax(nap_NN),valid_min = np.nanmin(nap_NN[nap_NN>-990]))
        fill_atributes(cdom_NN_var,long_name = 'Neural Network estimate of the logarithm of the concentration of Colored Dissolved Organic Matter', axis=False,units='log [mg m^(-3)]',valid_max = np.nanmax(cdom_NN),valid_min = np.nanmin(cdom_NN[cdom_NN>-990]))
    if calc_chla:
        fill_atributes(chla_var,long_name = 'Bayesian estimate of the logarithm of the concentration of Chlorophyll', axis=False,units='log [mg m^(-3)]',valid_max = np.nanmax(chla),valid_min = np.nanmin(chla[chla>-990]))
        fill_atributes(nap_var,long_name = 'Bayesian estimate of the logarithm of the concentration of Non Algal Particles', axis=False,units='log [mg m^(-3)]',valid_max = np.nanmax(nap),valid_min = np.nanmin(nap[nap>-990]))
        fill_atributes(cdom_var,long_name = 'Bayesian estimate of the logarithm of the concentration of Colored Dissolved Organic Matter', axis=False,units='log [mg m^(-3)]',valid_max = np.nanmax(cdom),valid_min = np.nanmin(cdom[cdom>-990]))
        fill_atributes(dchla_var,long_name = 'Uncertainty for the bayesian estimate of the logarithm of the concentration of Chlorophyll', axis=False,units='log [mg m^(-3)]',valid_max = np.nanmax(dchla),valid_min = np.nanmin(dchla[dchla>-990]))
        fill_atributes(dnap_var,long_name = 'Uncertainty for the bayesian  estimate of the logarithm of the concentration of Non Algal Particles', axis=False,units='log [mg m^(-3)]',valid_max = np.nanmax(dnap),valid_min = np.nanmin(dnap[dnap>-990]))
        fill_atributes(dcdom_var,long_name = 'Uncertainty for the bayesian  estimate of the logarithm of the concentration of Colored Dissolved Organic Matter', axis=False,units='log [mg m^(-3)]',valid_max = np.nanmax(dcdom),valid_min = np.nanmin(dcdom[dcdom>-990]))
    if calc_kd:
        fill_atributes(kd_var,long_name = 'Bayesian derivation of downward light attenuation coefficient', axis=False,units='m^(-1)',valid_max = np.nanmax(kd),valid_min = np.nanmin(kd[kd>-990]))
        fill_atributes(dkd_var,long_name = 'Uncertainty for the bayesian downward light attenuation coefficient', axis=False,units='m^(-1)',valid_max = np.nanmax(dkd),valid_min = np.nanmin(dkd[dkd>-990]))
    if calc_bbp:
        fill_atributes(bbp_var,long_name = 'Bayesian derivation of particulate backward scattering coefficient', axis=False,units='m^(-1)',valid_max = np.nanmax(bbp),valid_min = np.nanmin(bbp[bbp>-990]))
        fill_atributes(dbbp_var,long_name = 'Uncertainty for the particulate backward scattering coefficient', axis=False,units='m^(-1)',valid_max = np.nanmax(dbbp),valid_min = np.nanmin(dbbp[dbbp>-990]))
    if calc_rrs: fill_atributes(RRS_var,long_name = 'Forward computation of the Remote Sensing Reflectance', axis=False,units='-',valid_max = np.nanmax(RRS),valid_min = np.nanmin(RRS[RRS>-990]))
                   
    time_var[:] = (datetime(year=2020,month=1,day=1) - datetime(year=1981,month=1,day=1)).days
    lat_var[:] = lat
    lon_var[:] = lon
    lambda_var[:] = np.array([412.5,442.5,490.0,510.0,555.0])

    if calc_NN:
        chla_NN_var[0,:,:] = chla_NN
        nap_NN_var[0,:,:] = nap_NN
        cdom_NN_var[0,:,:] = cdom_NN
        del chla_NN
        del nap_NN
        del cdom_NN
        
    if calc_chla:
        chla_var[0,:,:] = chla
        dchla_var[0,:,:] = dchla
        nap_var[0,:,:] = nap
        dnap_var[0,:,:] = dnap
        cdom_var[0,:,:] = cdom
        dcdom_var[0,:,:] = dcdom        
        del chla
        del dchla
        del nap
        del dnap
        del cdom
        del dcdom

    if calc_kd:
        kd_var[0,:,:,:] = kd
        dkd_var[0,:,:,:] = dkd
        del kd
        del dkd
        
    if calc_bbp:
        bbp_var[0,:,:,:] = bbp
        dbbp_var[0,:,:,:] = dbbp
        del bbp
        del dbbp
        
    if calc_rrs:
        RRS_var[0,:,:,:] = RRS
        del RRS
        
    del lat
    del lon
    
    ncfile.close()

    
if __name__ == '__main__':

    print('modules loaded')
    conf_ = read_parameters()
    comm,rank,nranks = set_mpi('127.0.0.1','29500',1)
    

    #Im using mpi to compute multiple points at the same time, each with different cores. 
    #each core knows where the data is and the hyperparameter values. 
    scratch_path = conf_['scratch_path']
    perturbation_factors = tensor(np.load(conf_['perturbation_factors_path']))[-100:].mean(axis=0).to(float32)
    my_device = conf_['device']
    constant = read_constants(file1=conf_['constants_path']+'/cte_lambda.csv',file2=conf_['constants_path']+'/cte.csv',my_device = my_device)
    date_str = conf_['date']

    ##
    date_str = '20080605'
    conf_['rrs_data_path'] = rrs_data_file = '/g100_scratch/userexternal/csotolop/DIIM_output/RRS/20080605_cmems_obs-oc_med_bgc-reflectance_my_l3-multi-1km_P1D.nc'
    conf_['oasim_data_path'] = oasim_data_path = '/g100_scratch/userexternal/csotolop/DIIM_output/OASIM_maps/oasim_map_20080605.nc'
    conf_['dateformat'] = '%Y%m%d'
    conf_['date'] = date_str
    date = datetime.strptime(date_str,conf_['dateformat'])
    time_utc = datetime(year = date.year,month = date.month,day=date.day,hour=11)
    output_data_path = conf_['output_path'] = '/g100_scratch/userexternal/csotolop/DIIM_output/diim_maps'+ '/diim_map_' + date_str + '.nc'
    ##

    dataset = inversion(dateformat=conf_['dateformat'],scratch_path=conf_['scratch_path'],oasim_data_path=conf_['oasim_data_path'],rrs_data_path=conf_['rrs_data_path'],PAR_path=conf_['PAR_path'],zenith_path=conf_['zenith_path'],date_str=conf_['date'],rank=rank,nranks=nranks,comm=comm,my_device=my_device,perturbation_factors = perturbation_factors,constant=constant)


    
    ##
    #if rank == 0 :
    #    dataset = read_map_class(oasim_data_file=oasim_data_path,\
    #                             rrs_data_file=rrs_data_file,time_utc = time_utc,scratch_path = conf_['scratch_path'] ,\
    #                             PAR_folder = conf_['PAR_path'],zenith_folder = conf_['zenith_path'],store_data = True,dateformat = conf_['dateformat'])
    ##
    
    if rank == 0:
        create_map(conf_['scratch_path'],dataset,output_path = conf_['output_path'],date_str = conf_['date'])

    

    





