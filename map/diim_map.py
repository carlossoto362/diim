

import netCDF4
import numpy as np
import torch
from torch.utils.data import DataLoader
#from cartopy import config
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import time
import sys
import os
from pvlib.solarposition import get_solarposition
import pvlib
from datetime import datetime, timedelta
from pandas import DatetimeIndex
from tqdm import tqdm
#from multiprocessing.pool import Pool
import diimpy.Forward_module as fm
import diimpy.read_data_module as rdm
import diimpy.bayesian_inversion as bayes
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
    print('Zenith angle computed in time:',time.time() - time_init)
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
    print('PAR computed in time:',time.time() - time_init)
    return PAR_

class read_map_class():
    def __init__(self,oasim_data_file = '/path/where/to/read/oasim/data',rrs_data_file='/path/where/to/read/rrs/data',\
                 time_utc = datetime(year=2020,month=3,day=1),PAR_folder = '/path/where/to/read/PAR',\
                 zenith_folder = '/path/where/to/read/zenith/angle',my_precision = torch.float32,\
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
            
            OASIM_data = netCDF4.Dataset(oasim_data_file)
            RRS_data = netCDF4.Dataset(rrs_data_file)
            
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
            
            x_data[:,10] = torch.ones((n_points)) * 412.5
            x_data[:,11] = torch.ones((n_points)) * 442.5
            x_data[:,12] = torch.ones((n_points)) * 490
            x_data[:,13] = torch.ones((n_points)) * 510
            x_data[:,14] = torch.ones((n_points)) * 555
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

            nan_y_points = np.all(y_data<-900,axis=1)
            nan_x_points = np.all(x_data<-900,axis=1)
            nan_points_ = nan_y_points | nan_x_points
            nan_points = np.where(nan_points_ == True)[0]
            non_nan_points = np.where(nan_points_ == False)[0]

            general_parameters['input_data_path'] = [scratch_path + '/input_data']
            if not os.path.exists(scratch_path + '/input_data'): os.mkdir(scratch_path + '/input_data')
            np.save(scratch_path + '/input_data/x_data_'+self.date_str+'.npy',x_data[non_nan_points])
            np.save(scratch_path + '/input_data/y_data_'+self.date_str+'.npy',y_data[non_nan_points])
            
            general_parameters['non_nan_points_path'] = [scratch_path + '/non_nan_points_'+self.date_str+'.npy']
            np.save(scratch_path + '/non_nan_points'+self.date_str+'.npy',non_nan_points)
            general_parameters['non_nan_points_len'] = [len(non_nan_points)]

            
            general_parameters.to_csv(scratch_path + '/general_parameters_'+self.date_str+'.csv')
            
            y_data[y_data < -900.] = np.nan
            x_data[x_data < -900.] = np.nan
            self.x_data = torch.tensor(x_data).to(my_precision)
            self.y_data = torch.tensor(y_data).to(my_precision)
            self.non_nan_points = non_nan_points
            read_general_parameters(self,scratch_path + '/general_parameters_'+self.date_str+'.csv')
            
        elif os.path.exists(scratch_path + '/general_parameters_'+self.date_str+'.csv'):
            
            read_general_parameters(self,scratch_path + '/general_parameters_'+self.date_str+'.csv')
            y_data = np.load(self.input_data_path + '/y_data_'+self.date_str+'.npy')
            x_data = np.load(self.input_data_path + '/x_data_'+self.date_str+'.npy')
            y_data[y_data < -900.] = np.nan
            x_data[x_data < -900.] = np.nan
            self.x_data = torch.tensor(x_data).to(my_precision)
            self.y_data = torch.tensor(y_data).to(my_precision)
            self.non_nan_points = torch.tensor(np.load(self.non_nan_points_path))
            
        else:
            print(scratch_path + "/general_parameters_"+dateformat+".csv dosn't exist,")
            print("please enter the scratch_path where the input data and general_parameters_"+dateformat+".csv was stores,")
            print("or use the parameter 'store_data = True'")
        #print('data loaded..., time: ',time.time() - init_time )
        

    def __len__(self):
        return self.non_nan_points_len

    
    def __getitem__(self, idx):
        image = torch.empty((5,5))
        image[:,0] = self.x_data[self.non_nan_points][idx][:5]
        image[:,1] = self.x_data[self.non_nan_points][idx][5:10]
        image[:,2] = self.x_data[self.non_nan_points][idx][10:15]
        image[:,3] = self.x_data[self.non_nan_points][idx][15]
        image[:,4] = self.x_data[self.non_nan_points][idx][16]

        label = self.y_data[self.non_nan_points][idx]

        return image,label
    
    def get_coordinates(self):
        mesh_lat= np.load(self.mesh_lat_path)
        mesh_lon = np.load(self.mesh_lon_path)
        return mesh_lat,mesh_lon

    

def local_initial_conditions_nn(F_model,constant,data,precision = torch.float32,my_device = 'cpu',rearange_needed=True):
    
    init_time_ = time.time()
    best_result_config = torch.load(MODEL_HOME + '/settings/VAE_model/model_second_part_final_config.pt')

    number_hiden_layers_mean = best_result_config['number_hiden_layers_mean']
    dim_hiden_layers_mean = best_result_config['dim_hiden_layers_mean']
    dim_last_hiden_layer_mean = best_result_config['dim_last_hiden_layer_mean']
    alpha_mean = best_result_config['alpha_mean']
    number_hiden_layers_cov = best_result_config['number_hiden_layers_cov']
    dim_hiden_layers_cov = best_result_config['dim_hiden_layers_cov']
    dim_last_hiden_layer_cov = best_result_config['dim_last_hiden_layer_cov']
    alpha_cov = best_result_config['alpha_cov']
    x_mul = torch.tensor(best_result_config['x_mul']).to(precision).to(my_device)
    x_add = torch.tensor(best_result_config['x_add']).to(precision).to(my_device)
    y_mul = torch.tensor(best_result_config['y_mul']).to(precision).to(my_device)
    y_add = torch.tensor(best_result_config['y_add']).to(precision).to(my_device)

    model_NN = bayes.NN_second_layer(output_layer_size_mean=3,number_hiden_layers_mean = number_hiden_layers_mean,\
                           dim_hiden_layers_mean = dim_hiden_layers_mean,alpha_mean=alpha_mean,dim_last_hiden_layer_mean = dim_last_hiden_layer_mean,\
                           number_hiden_layers_cov = number_hiden_layers_cov,\
                           dim_hiden_layers_cov = dim_hiden_layers_cov,alpha_cov=alpha_cov,dim_last_hiden_layer_cov = dim_last_hiden_layer_cov,x_mul=x_mul,x_add=x_add,\
                           y_mul=y_mul,y_add=y_add,constant = constant,model_dir = MODEL_HOME + '/settings/VAE_model').to(my_device)


    model_NN.load_state_dict(torch.load(MODEL_HOME + '/settings/VAE_model/model_second_part_chla_centered.pt'))
    model_NN.eval()


    if rearange_needed == True:
        X,Y = data
        data_rearanged = torch.empty((X.shape[0],1,17))
        data_rearanged[:,0,:5] = Y
        data_rearanged[:,0,5:10] = X[:,0]
        data_rearanged[:,0,10:15] = X[:,1]
        data_rearanged[:,0,15] = X[:,3,0]
        data_rearanged[:,0,16] = X[:,4,0]
    else:
        data_rearanged = data
        
    z_hat,cov_z,mu_z,kd_hat,bbp_hat,rrs_hat = model_NN(data_rearanged) #we are working with \lambda as imput, but the NN dosent use it.
    mu_z = (mu_z* model_NN.y_mul[0] + model_NN.y_add[0]).clone().detach()

    state_dict = F_model.state_dict()
    state_dict['chparam']  = mu_z.unsqueeze(1)
    F_model.load_state_dict(state_dict)
    print('time for NN data,', time.time() - init_time_)
    return mu_z

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
    return parser  

    
if __name__ == '__main__':

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

    
    torch.set_num_threads(1)

    #Im using mpi to compute multiple points at the same time, each with different cores. 
    comm = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nranks = comm.size
    #in case the code try to be run in GPU, lets specify master as local host. 
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    #each core knows where the data is and the hyperparameter values. 
    scratch_path = conf_['scratch_path']
    perturbation_factors = torch.tensor(np.load(conf_['perturbation_factors_path']))[-100:].mean(axis=0).to(torch.float32)
    my_device = conf_['device']
    constant = rdm.read_constants(file1=conf_['constants_path']+'/cte_lambda.csv',file2=conf_['constants_path']+'/cte.csv',my_device = my_device)
    dataset = read_map_class(oasim_data_file=conf_['oasim_data_path'],\
                             rrs_data_file=conf_['rrs_data_path'],time_utc = datetime(year=2020,month=1,day=2,hour=11),scratch_path = scratch_path + '/data_to_read',\
                             PAR_folder = conf_['PAR_path'],zenith_folder = conf_['zenith_path'],store_data = True,dateformat = conf_['dateformat'])
    
    lr = 0.029853826189179603
    x_a = torch.zeros(3)
    s_a = torch.eye(3) * 4.9
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf

    batch_size = 2000#len(dataset)
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

    if rank == 0:
        print('starting to process the map with {} points, by grouping them in groups of {}, and working with {} simultaneus processes'.format(len(dataset.x_data),batch_size,n_workers))
    comm.Barrier()
    
    for i,data in itertools.islice(enumerate(dataloader),worker_i,None,lenght_iter):

        if (nranks == 1) or (rank != 0):
            
            len_batch = len(data[0])
            if nranks == 1:

                print('Working iterations {} to {} from {}'.format(batch_size*i,batch_size*i + len_batch,len(dataset)))
            model = fm.Forward_Model(num_days=len_batch).to(my_device)
            model.perturbation_factors = perturbation_factors
            chla_NN_sen = local_initial_conditions_nn(model,constant,data,precision = torch.float32,my_device = 'cpu').numpy()
            loss = fm.RRS_loss(x_a,s_a,s_e,num_days=len_batch,my_device = my_device)
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            output = bayes.train_loop(data,model,loss,optimizer,4000,kind='all',\
                                      num_days=len_batch,constant = constant,perturbation_factors_ = perturbation_factors, scheduler = True)

            chla_log_sen = output['X_hat']
            kd_log_sen = output['kd_hat']
            bbp_log_sen = output['bbp_hat']
            RRS_log_sen = output['RRS_hat']

            if nranks!=1:
                comm.Send([chla_NN_sen.astype(np.float32), MPI.FLOAT],dest=0,tag=1)
                comm.Send([chla_log_sen.astype(np.float32), MPI.FLOAT],dest=0,tag=2)
                comm.Send([kd_log_sen.numpy().astype(np.float32), MPI.FLOAT],dest=0,tag=3)
                comm.Send([bbp_log_sen.numpy().astype(np.float32), MPI.FLOAT],dest=0,tag=4)
                comm.Send([RRS_log_sen.numpy().astype(np.float32), MPI.FLOAT],dest=0,tag=5)
            else:
                chla_NN_rec[batch_size*i:batch_size*i + len_batch] = chla_NN_sen
                chla_log_rec[batch_size*i:batch_size*i + len_batch] = chla_log_sen
                kd_log_rec[batch_size*i:batch_size*i + len_batch] = kd_log_sen
                bbp_log_rec[batch_size*i:batch_size*i + len_batch] = bbp_log_sen
                RRS_log_rec[batch_size*i:batch_size*i + len_batch] = RRS_log_sen
                
        if (rank == 0) & (nranks!=1):
            len_batch = len(data[0])
            rank_of_worker_sender = (i%n_workers) + 1
            comm.Recv([ chla_NN_rec[batch_size*i:batch_size*i + len_batch],MPI.FLOAT],source=rank_of_worker_sender,tag=1 )
            comm.Recv([ chla_log_rec[batch_size*i:batch_size*i + len_batch],MPI.FLOAT],source=rank_of_worker_sender,tag=2 )
            comm.Recv([ kd_log_rec[batch_size*i:batch_size*i + len_batch],MPI.FLOAT],source=rank_of_worker_sender,tag=3 )
            comm.Recv([ bbp_log_rec[batch_size*i:batch_size*i + len_batch],MPI.FLOAT],source=rank_of_worker_sender,tag=4 )
            comm.Recv([ RRS_log_rec[batch_size*i:batch_size*i + len_batch],MPI.FLOAT],source=rank_of_worker_sender,tag=5 )
            print('communication received from rank {}, corresponding to elements from {} to {}'.format(rank_of_worker_sender,batch_size*i,batch_size*i + len_batch))


    if rank == 0:
        np.save(conf_['output_path']+'/chla_NN.npy',chla_NN_rec)
        np.save(conf_['output_path']+'/chla_log.npy',chla_log_rec)
        np.save(conf_['output_path']+'/kd_log.npy',kd_log_rec)
        np.save(conf_['output_path']+'/bbp_log.npy',bbp_log_rec)
        np.save(conf_['output_path']+'/RRS_log.npy',RRS_log_rec)
    
    #chla_NN_rec = np.empty((dataset.len_lat*dataset.len_lon,3))
    #chla_log_rec = np.empty((dataset.len_lat*dataset.len_lon,6))
    #kd_log_rec = np.empty((dataset.len_lat*dataset.len_lon,10))
    #bbp_log_rec = np.empty((dataset.len_lat*dataset.len_lon,6))
    #RRS_log_rec = np.empty((dataset.len_lat*dataset.len_lon,5))








#plot_map(map_=Edif_555,ncfile = OASIM_data,map_flag = True)

#Edir = torch.mean(torch.tensor(OASIM_data['edout'][:,:,:,:]), axis = 0).to(my_precision)
#print(OASIM_data['wavelength'][:])




    





