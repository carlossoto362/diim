

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
import diimpy.bayesian_inversion as bayes
import diimpy.Forward_module as fm
import diimpy.read_data_module as rdm
import pandas as pd
from mpi4py import MPI
import itertools


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
    

%njit
def linear_splines(x,xs,map_):
    
    position = 0
    for i in range(1,len(xs)-1):
        if (x<=xs[i]) and (x>xs[i-1]):
            return map_[i-1] + (map_[i] - map_[i-1])*(x - xs[i-1])/(xs[i] - xs[i-1])

%njit        
def get_solarposition_(time_utc,lats,lons):
    year = time_utc.year
    time_elapsed = time_utc  - datetime(year = year,month=1,day=1)
    hour_ = time_utc.time()
    
    days_N = time_elapsed.days + time_elapsed.seconds/86400 #86400 = 60*60*24
    hour_angle =  (hour_.hour + hour_.minute/60 + hour_.second/3600  + (lons*12/180) - 12)*np.pi/12 #local time, (lons*12/180) is degrees to hours
    sun_declination = - np.arcsin(
         0.39779 * np.cos(  (0.98565*(days_N + 10) + 1.914*np.sin(  0.98565*(days_N - 2)*np.pi/180  ))*np.pi/180 ) #
    )
    zenith_angle = np.arccos(
        np.sin(lats*np.pi/180)*np.sin(sun_declination) + np.cos(lats*np.pi/180)*np.cos(sun_declination)*np.cos(hour_angle)
    )
    return zenith_angle

%njit
def zenith_angle(time_utc,lats,lons,folder = '.'):
    np.save(folder + '/zenith_'+time.strftime('%Y%m%d_%H:%M%S.npy'),get_solar_position_(time_utc,lats,lons))
    return zenith_

%njit
def PAR_calculator(wl,Edir,Edif,time,folder = '.'):

    if os.path.exists(folder + '/PAR_'+time.strftime('%Y%m%d.npy')):
        PAR_ = np.load(folder + '/PAR_'+time.strftime('%Y%m%d.npy'))
        return PAR_

    Na = 6.0221408e+23
    h = 6.62607015e-34
    c = 299792458 
    constant = (10**6)/ (Na*h*c)
    PAR_ = np.zeros((Edir.shape[1:]))
    irange = ((wl>401) & (wl<699))
    ilen = len(wl[irange])

    for i in range(ilen):
        PAR_ += (Edif[irange][i] + Edir[irange][i])*wl[irange][i]*1e-9 # Edif are the bin irradiance, the integral Edif for a rectangle of size dlambda

    PAR_ = PAR_  * constant

    np.save(folder + '/PAR_'+time.strftime('%Y%m%d.npy'),PAR_)
    return PAR_

class read_map_class():
    def __init__(self,oasim_data_file = '/path/where/to/read/oasim/data',rrs_data_file='/path/where/to/read/rrs/data',\
                 time_str = '20200101 10:00:00',PAR_folder = '/path/where/to/read/PAR',\
                 zenith_folder = '/path/where/to/read/zenith/angle',my_precision = torch.float32,\
                 scratch_path = '/path/to/scratch/',store_data = False):
        
        init_time = time.time()
        self.my_precision = my_precision

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
            
            wl = torch.tensor(OASIM_data['wavelength'][:]).to(my_precision)
            
            general_parameters['len_lat']  = [len(OASIM_data['lat'][:])]
            general_parameters['len_lon']  = [len(OASIM_data['lon'][:])]
            
            general_parameters['n_points']  = [len(OASIM_data['lat'][:])*len(OASIM_data['lon'][:])]
            n_points = int(general_parameters['n_points'].iloc[0])
            
            mesh_lat,mesh_lon = np.meshgrid(OASIM_data['lat'][:],OASIM_data['lon'][:])
            mesh_lat, mesh_lon = mesh_lat.T, mesh_lon.T
            general_parameters['mesh_lat_path'] = [scratch_path + '/mesh_lat.pt']
            torch.save(torch.tensor(mesh_lat),scratch_path + '/mesh_lat.pt')
            general_parameters['mesh_lon_path'] = [scratch_path + '/mesh_lon.pt']
            torch.save(torch.tensor(mesh_lon),scratch_path + '/mesh_lon.pt')
        
            x_data = torch.empty((n_points,17))
            
            Edif = torch.mean(torch.tensor(OASIM_data['esout'][:,:,:,:]), axis = 0).to(my_precision)
            x_data[:,0] = linear_splines(412.5,wl,Edif).reshape((n_points))
            x_data[:,1] = linear_splines(442.5,wl,Edif).reshape((n_points))
            x_data[:,2] = linear_splines(490,wl,Edif).reshape((n_points))
            x_data[:,3] = linear_splines(510,wl,Edif).reshape((n_points))
            x_data[:,4] = linear_splines(555,wl,Edif).reshape((n_points))
            
            Edir = torch.mean(torch.tensor(OASIM_data['edout'][:,:,:,:]), axis = 0).to(my_precision)
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
            time_ = datetime.strptime(time_str ,'%Y%m%d %H:%M:%S')
            general_parameters['time'] = [time_str]
            general_parameters['time_format'] = ['%Y%m%d %H:%M:%S']
            x_data[:,15] = zenith_angle(time_,OASIM_data['lat'][:],OASIM_data['lon'][:],folder = zenith_folder).reshape((n_points))
            
            x_data[:,16] = PAR_calculator(wl,Edir,Edif,time_,folder = PAR_folder).reshape((n_points))
            
            y_data = torch.empty((n_points),5).to(my_precision)
            y_data[:,0] = torch.tensor(RRS_data['RRS412'][0,:,:].reshape((n_points)))
            y_data[:,1] = torch.tensor(RRS_data['RRS443'][0,:,:].reshape((n_points)))
            y_data[:,2] = torch.tensor(RRS_data['RRS490'][0,:,:].reshape((n_points)))
            y_data[:,3] = torch.tensor(RRS_data['RRS510'][0,:,:].reshape((n_points)))
            y_data[:,4] = torch.tensor(RRS_data['RRS555'][0,:,:].reshape((n_points)))
            
            y_data[y_data < -900.] = np.nan
            x_data[x_data < -900.] = np.nan

            nan_y_points = np.unique( np.argwhere( y_data.isnan() )[0,:] )
            nan_x_points = np.unique( np.argwhere( x_data.isnan() )[0,:] )
            nan_points = np.unique(np.concatenate([nan_y_points,nan_x_points]))
            non_nan_points = np.arange(n_points)[~np.isin(np.arange(n_points),nan_points)]

            general_parameters['input_data_path'] = [scratch_path + '/input_data']
            if not os.path.exists(scratch_path + '/input_data'): os.mkdir(scratch_path + '/input_data')
            torch.save(torch.tensor(x_data[non_nan_points]),scratch_path + '/input_data/x_data.pt')
            torch.save(torch.tensor(y_data[non_nan_points]),scratch_path + '/input_data/y_data.pt')
            
            general_parameters['non_nan_points_path'] = [scratch_path + '/non_nan_points.pt']
            torch.save(torch.tensor(non_nan_points),scratch_path + '/non_nan_points.pt')
            general_parameters['non_nan_points_len'] = [len(non_nan_points)]

            general_parameters.to_csv(scratch_path + '/general_parameters.csv')            
            
        if os.path.exists(scratch_path + '/general_parameters.csv'):
            read_general_parameters(self,scratch_path + '/general_parameters.csv')
            self.x_data = torch.load(self.input_data_path + '/x_data.pt')
            self.y_data = torch.load(self.input_data_path + '/y_data.pt')
        else:
            print(scratch_path + "/general_parameters.csv dosn't exist,")
            print("please enter the scratch_path where the input data and general_parameters.csv was stores,")
            print("or use the parameter 'store_data = True'")
        print('data loaded..., time: ',time.time() - init_time )
        

    def __len__(self):
        return self.non_nan_points_len

    
    def __getitem__(self, idx):
        image = torch.empty((5,5))
        image[:,0] = self.x_data[idx][:5]
        image[:,1] = self.x_data[idx][5:10]
        image[:,2] = self.x_data[idx][10:15]
        image[:,3] = self.x_data[idx][15]
        image[:,4] = self.x_data[idx][16]

        label = self.y_data[idx]

        return image,label


        return label,image
    def get_coordinates(self):
        mesh_lat= torch.load(self.mesh_lat_path)
        mesh_lon = torch.load(self.mesh_lon_path)
        return mesh_lat,mesh_lon

    

def local_initial_conditions_nn(F_model,constant,data,precision = torch.float32,my_device = 'cpu',rearange_needed=True):
    
    init_time_ = time.time()
    best_result_config = torch.load(MODEL_HOME + '/VAE_model/model_second_part_final_config.pt')

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
                           y_mul=y_mul,y_add=y_add,constant = constant,model_dir = MODEL_HOME + '/VAE_model').to(my_device)


    model_NN.load_state_dict(torch.load(MODEL_HOME + '/VAE_model/model_second_part_chla_centered.pt'))
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
    state_dict['chparam'] = mu_z.unsqueeze(1)
    F_model.load_state_dict(state_dict)
    print('time for NN data,', time.time() - init_time_)
    return mu_z

def create_nan_array(shape):
        nan_array = np.empty(shape)
        nan_array[:] = np.nan
        return nan_array

    
if __name__ == '__main__':
    #the one d model is simple, and using multiple cores for the inference is slower than using only one core. 
    torch.set_num_threads(1)

    #Im using mpi to compute multiple points at the same time, each with different cores. 
    comm = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nranks = comm.size
    #in case the code try to be run in GPU, lets specify master as local host. 
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    #each core nows kwhere the data is and the hyperparameter values. 
    scratch_path = '/g100_scratch/userexternal/csotolop/results_one_d_model'
    perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/plot_data/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy'))[-100:].mean(axis=0).to(torch.float32)
    my_device = 'cpu'
    constant = rdm.read_constants(file1=MODEL_HOME + '/cte_lambda.csv',file2=MODEL_HOME + '/cst.csv',my_device = my_device)
    dataset = read_map_class(oasim_data_file='/g100/home/userexternal/csotolop/maps/OASIM_MED_20200102_07:00:00-20200102_16:0000.nc',\
                             rrs_data_file='/g100/home/userexternal/csotolop/maps/RRS_med_20200102.nc',time_str = '20200101 10:00:00',scratch_path = scratch_path + '/data_to_read',\
                             PAR_folder = '/g100/home/userexternal/csotolop/maps',zenith_folder = '/g100/home/userexternal/csotolop/maps',store_data = False)
    
    lr = 0.029853826189179603
    x_a = torch.zeros(3)
    s_a = torch.eye(3) * 4.9
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf

    batch_size = 500#len(dataset)
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
        np.save('/g100/home/userexternal/csotolop/maps/chla_NN.npy',chla_NN_rec)
        np.save('/g100/home/userexternal/csotolop/maps/chla_log.npy',chla_log_rec)
        np.save('/g100/home/userexternal/csotolop/maps/kd_log.npy',kd_log_rec)
        np.save('/g100/home/userexternal/csotolop/maps/bbp_log.npy',bbp_log_rec)
        np.save('/g100/home/userexternal/csotolop/maps/RRS_log.npy',RRS_log_rec)
    
    #chla_NN_rec = np.empty((dataset.len_lat*dataset.len_lon,3))
    #chla_log_rec = np.empty((dataset.len_lat*dataset.len_lon,6))
    #kd_log_rec = np.empty((dataset.len_lat*dataset.len_lon,10))
    #bbp_log_rec = np.empty((dataset.len_lat*dataset.len_lon,6))
    #RRS_log_rec = np.empty((dataset.len_lat*dataset.len_lon,5))








#plot_map(map_=Edif_555,ncfile = OASIM_data,map_flag = True)

#Edir = torch.mean(torch.tensor(OASIM_data['edout'][:,:,:,:]), axis = 0).to(my_precision)
#print(OASIM_data['wavelength'][:])




    





