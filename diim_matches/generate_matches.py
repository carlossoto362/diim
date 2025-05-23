from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
import sys
import os
import pandas as pd
from tqdm import tqdm
from diimpy import oasim_map as oasim
from diimpy import diim_map as dm
from bitsea.commons.mask import Mask
from mpi4py import MPI
import torch
from torch.utils.data import DataLoader
from diimpy import Forward_module as fm
from diimpy import read_data_module as rdm
import itertools
import time
import matplotlib.pyplot as plt

def save_matches(save=True):
    k = 0
    rrs_path = '/g100_work/OGS23_PRACE_IT/csoto/rrs_data/V11C/SAT/DAILY/CHECKED'
    chla_path = '/g100_scratch/userexternal/vdibiagi/EMODnet_2022/NEW_int/fromSC/publication/Zenodo/DEFINITIVO/MedBGCins_nut.nc'
    chla_path = '/g100_scratch/userexternal/vdibiagi/EMODnet_2022/NEW_int/fromSC/publication/ZenodoUpdated/toBePublished/MedBGCins_nut_v1.nc'

    data = nc.Dataset(chla_path)
    depths = data['DATA'][5]
    depths_flag = (depths <= 0)
    chla_data = data['DATA'][-3]
    chla_data.mask |= depths_flag
    years = data['DATA'][0]
    months = data['DATA'][1]
    days = data['DATA'][2]
    len_all = len(chla_data)

    lats = data['DATA'][3]
    lons = data['DATA'][4]

    rrs_mask = np.ones(len_all)
    rrs_mask = (rrs_mask<0)

    rrs_labels = ['RRS412','RRS443','RRS490','RRS510','RRS555']
    rrs_subsampled = np.zeros((len_all,5))
    rrs_std = np.zeros((len_all,5))

    pixels_surrounding = 2
    dates = []

    for i in tqdm(range(len_all)):
        
        dates.append(datetime(year = int(years.compressed()[i]),month = int(months.compressed()[i]),day=int(days.compressed()[i])))
        #if (lons[i] <7) | (lons[i] >10) | (lats[i] <44) | (lats[i] > 45):
        #    rrs_mask[i] = True
        #    continue
        if chla_data.mask[i] == True:
            continue
    
        date_str = dates[i].strftime('%Y%m%d_cmems_obs-oc_med_bgc-reflectance_my_l3-multi-1km_P1D.nc')
        rrs_data_path = rrs_path + '/' + date_str
        if ~np.array(os.path.exists(rrs_data_path)):
            rrs_mask[i] = True
            continue
    
        rrs_data = nc.Dataset(rrs_data_path)
        

        lat_inf = np.argwhere( (rrs_data['lat'] < lats[i]) )[-1]
        lat_sup = np.argwhere( (rrs_data['lat'] > lats[i]) )[0]
        if lat_inf >= pixels_surrounding:
            lat_range_inf = lat_inf - pixels_surrounding
        else:
            lat_range_inf = 0
        if lat_sup < len(rrs_data['lat']) - pixels_surrounding:
            lat_range_sup = lat_sup + pixels_surrounding
        else:
            lat_range_sup = -1
    
        lon_inf = np.argwhere( (rrs_data['lon'] < lons[i]) )[-1]
        lon_sup = np.argwhere( (rrs_data['lon'] > lons[i]) )[0]
    
        if lon_inf >= pixels_surrounding:
            lon_range_inf = lon_inf - pixels_surrounding
        else:
            lon_range_inf = 0
        if lon_sup < len(rrs_data['lon']) - pixels_surrounding:
            lon_range_sup = lon_sup + pixels_surrounding
        else:
            lon_range_sup = -1


        for j,rrs_label in enumerate(rrs_labels):

            rrs_range = rrs_data[rrs_label][0,int(lat_range_inf):int(lat_range_sup),int(lon_range_inf):int(lon_range_sup)]
            if len(rrs_range.compressed())<((pixels_surrounding*2+1)**2)/2:
                rrs_mask[i] = True
        
            rrs_subsampled[i,j] = np.ma.mean(rrs_range)
            
            rrs_std[i,j] = np.ma.std(rrs_range)
            if (rrs_std[i,j]/rrs_subsampled[i,j]) > 0.15:
                rrs_mask[i] = True
                
        #if rrs_mask[i] != True:
        #    print(np.nanmean(rrs_data['RRS412'][0,int(lat_range_inf):int(lat_range_sup),int(lon_range_inf):int(lon_range_sup)]))
        #    print(np.nanmean(rrs_data['RRS555'][0,int(lat_range_inf):int(lat_range_sup),int(lon_range_inf):int(lon_range_sup)]))
        #    print(rrs_subsampled[i])
        #    k+=1
        #if k == 100:
        #    break

    chla_data.mask |= rrs_mask
    lats.mask = chla_data.mask
    lons.mask = chla_data.mask

    rrs_mask = np.ones((len_all,5))
    rrs_mask = (rrs_mask<0)

    for i in range(rrs_mask.shape[0]):
        rrs_mask[i] = (rrs_mask[i] | chla_data.mask[i])
    
    rrs_subsampled = np.ma.array(rrs_subsampled,mask = rrs_mask)
    rrs_std = np.ma.array(rrs_std,mask = rrs_mask)
    dates = np.ma.array(dates,mask=chla_data.mask)

    if save == True:
        np.save('dates.npy',dates.data)
        np.save('rrs_subsampled.npy',rrs_subsampled.data)
        np.save('lats.npy',lats.data)
        np.save('lons.npy',lons.data)
        np.save('chla_data.npy',chla_data.data)

        np.save('rrs_std.npy',rrs_std.data)
        np.save('chla_mask.npy',chla_data.mask)
        np.save('rrs_mask.npy',rrs_subsampled.mask)
        
    return dates.data,rrs_subsampled.data,lats.data,lons.data,chla_data.data,rrs_std.data,chla_data.mask,rrs_subsampled.mask

class load_data_tensor_list():

    def __init__(self,rrs,par,zenith,edif,edir,wl,times,lons,lats,my_precision = torch.float32,my_device='cpu'):
        self.my_precision = my_precision
        self.my_device = my_device
        self.times = times
        
        self.x_column_names = ['Edif_412','Edif_442','Edif_490','Edif_510',\
                               'Edif_555','Edir_412','Edir_442','Edir_490','Edir_510','Edir_555','lambda_412','lambda_442',\
                               'lambda_490','lambda_510','lambda_555','zenith','PAR']
        
        self.y_column_names = ['RRS_412','RRS_442','RRS_490','RRS_510','RRS_555']

        self.n_points = len(self.times)

        self.x_data = torch.empty((self.n_points,17)).to(self.my_precision)
        self.x_data[:,0] = torch.tensor(dm.linear_splines(412.5,wl,edif.T))
        self.x_data[:,1] = torch.tensor(dm.linear_splines(442.5,wl,edif.T))
        self.x_data[:,2] = torch.tensor(dm.linear_splines(490,wl,edif.T))
        self.x_data[:,3] = torch.tensor(dm.linear_splines(510,wl,edif.T))
        self.x_data[:,4] = torch.tensor(dm.linear_splines(555,wl,edif.T))
        
        self.x_data[:,5] = torch.tensor(dm.linear_splines(412.5,wl,edir.T))
        self.x_data[:,6] = torch.tensor(dm.linear_splines(442.5,wl,edir.T))
        self.x_data[:,7] = torch.tensor(dm.linear_splines(490,wl,edir.T))
        self.x_data[:,8] = torch.tensor(dm.linear_splines(510,wl,edir.T))
        self.x_data[:,9] = torch.tensor(dm.linear_splines(555,wl,edir.T))

        self.x_data[:,10] = torch.ones((self.n_points)) * 412.5
        self.x_data[:,11] = torch.ones((self.n_points)) * 442.5
        self.x_data[:,12] = torch.ones((self.n_points)) * 490
        self.x_data[:,13] = torch.ones((self.n_points)) * 510
        self.x_data[:,14] = torch.ones((self.n_points)) * 555

        self.x_data[:,15] = torch.tensor(zenith)
        self.x_data[:,16] = torch.tensor(par)

        self.y_data = torch.tensor(rrs).to(self.my_precision)

        self.lats = lats
        self.lons = lons

        
    def __len__(self):
        return self.n_points

    def __getitem__(self, idx):
        image = torch.empty((5,5))
        image[:,0] = self.x_data[idx][:5]
        image[:,1] = self.x_data[idx][5:10]
        image[:,2] = self.x_data[idx][10:15]
        image[:,3] = self.x_data[idx][15]
        image[:,4] = self.x_data[idx][16]

        label = self.y_data[idx]

        return image,label
    
    def get_coordinate(self,idx):
        return self.lats[idx],self.lons[idx]
    
    def get_time(self,idx):
        return self.times[idx]

    
def diim_list(rrs,par,zenith,edif,edir,wl,times,lons,lats,constant=None,my_precision=torch.float32,my_device='cpu',\
              perturbation_factors = torch.tensor([0.99744277, 1.0072283 , 1.00813087, 1.00434172, 1.01976639,\
       0.98033676, 0.92478244, 0.98263952, 0.93829814, 1.0130856 ,\
       0.942826  , 0.96408048, 1.02222373, 0.58139362]),rank=0,nranks=1,save=True):
    
    dataset = load_data_tensor_list(rrs,par,zenith,edif,edir,wl,times,lons,lats,my_precision = my_precision,my_device=my_device)

    lr = 0.029853826189179603
    x_a = torch.zeros(3)
    s_a = torch.eye(3) * 1.13
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf

    batch_size = 1000
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

    comm.Barrier()
    init_time = time.time()
    for i,data in itertools.islice(enumerate(dataloader),worker_i,None,lenght_iter):
        
        if (nranks == 1) or (rank != 0):
            len_batch = len(data[0])
            
            model = fm.Forward_Model(num_days=len_batch).to(my_device)
            model.perturbation_factors = perturbation_factors
            chla_NN_sen = dm.local_initial_conditions_nn(model,constant,data,precision = my_precision,my_device = my_device).numpy()
            loss = fm.RRS_loss(x_a,s_a,s_e,num_days=len_batch,my_device = my_device)
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            output = dm.train_loop(data,model,loss,optimizer,4000,kind='all',\
                                      num_days=len_batch,constant = constant,perturbation_factors_ = perturbation_factors,calc_kd = False, calc_bbp = False)

            if np.isnan(output['X_hat']).all():
                model = fm.Forward_Model(num_days=len_batch).to(my_device)
                model.perturbation_factors = perturbation_factors
    
                loss = fm.RRS_loss(x_a,s_a,s_e,num_days=len_batch,my_device = my_device)
                optimizer = torch.optim.Adam(model.parameters(),lr=lr)
                output = dm.train_loop(data,model,loss,optimizer,4000,kind='all',\
                                       num_days=len_batch,constant = constant,perturbation_factors_ = perturbation_factors,calc_kd = False, calc_bbp = False)
                if np.isnan(output['X_hat']).all():
                    print(rank,i,'had a nan value for date, ',date_str)
            
            chla_log_sen = output['X_hat']
            RRS_log_sen = output['RRS_hat']

            if nranks!=1:
                comm.Send([chla_NN_sen.astype(np.float32), MPI.FLOAT],dest=0,tag=1)
                comm.Send([chla_log_sen.astype(np.float32), MPI.FLOAT],dest=0,tag=2)
                comm.Send([RRS_log_sen.numpy().astype(np.float32), MPI.FLOAT],dest=0,tag=5)

                del chla_NN_sen
                del chla_log_sen
                del RRS_log_sen
                del model
                del output
                del optimizer
                del loss

            else:
                chla_NN_rec[batch_size*i:batch_size*i + len_batch] = chla_NN_sen
                chla_log_rec[batch_size*i:batch_size*i + len_batch] = chla_log_sen
                RRS_log_rec[batch_size*i:batch_size*i + len_batch] = RRS_log_sen
                
                del chla_NN_sen
                del chla_log_sen
                del RRS_log_sen
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

            #print('communication received from rank {}, corresponding to elements from {} to {}'.format(rank_of_worker_sender,batch_size*i,batch_size*i + len_batch))
            
    if (rank == 0) & (save == True):
        np.save('chla_NN.npy',chla_NN_rec)
        np.save('chla_log.npy',chla_log_rec)
        np.save('RRS_log.npy',RRS_log_rec)
        return dataset,chla_NN_rec,chla_log_rec,RRS_log_rec






def generate_oasim_list(dates_start,dates_end,lats,lons,rank=0,nranks=1,save=False,read=False,data_path = './',prefix=''):

    dateformat = '%Y%m%d-%H:%M:%S'
    aerosol_datadir = '/g100_work/OGS_devC/NECCTON/OPTICS'
    cloud_datadir = '/g100_work/OGS_devC/NECCTON/OPTICS'
    atmosphere_prefix = 'atm'
    aerosol_prefix = 'aero'
    cloud_prefix = 'climatm'

    mask_file = '/g100_work/OGS_devC/V9C/RUNS_SETUP/PREPROC/MASK/meshmask.nc'
    wavelengths_file = '/g100_work/OGS23_PRACE_IT/csoto/DIIM/extern/OASIM_ATM/test/data/bin.txt'
    oasimlib_file = '/g100_work/OGS23_PRACE_IT/csoto/DIIM/map/oasim_map/liboasim-py.so'
    oasim_config_file = '/g100_work/OGS23_PRACE_IT/csoto/DIIM/map/oasim_map/oasim_config.yaml'

    TheMask = Mask.from_file(mask_file)
    partial_computation = pd.DataFrame(columns = ['index_','edout_','esout_','dates'])
    indexes_ = []
    edout_ = []
    esout_ = []
    dates_ = []
    if read == True:
        edout = np.load(data_path + '/' + prefix + '_edout.npy')
        esout = np.load(data_path + '/' + prefix + '_esout.npy') 
            

        return edout,esout

    for i in np.arange(len(dates_start))[rank::nranks]:
        atmosphere_datadir = dates_start[i].strftime('/g100_work/OGS_devC/NECCTON/OPTICS/%Y/%m')
        start_date = dates_start[i].strftime('%Y%m%d-%H:00:00')
        end_date = dates_end[i].strftime('%Y%m%d-%H:00:00')

        if dates_start[i] < datetime(year = 1999,month=1,day=1):
            indexes_.append(i)
            edout_.append(np.empty(33)*np.nan)
            esout_.append(np.empty(33)*np.nan)
            dates_.append(dates[i])
        else:
            oasim_output = oasim.getting_oasim_output(dateformat=dateformat,\
                                                      atmosphere_datadir=atmosphere_datadir,\
                                                      aerosol_datadir=aerosol_datadir,\
                                                      cloud_datadir=cloud_datadir,\
                                                      atmosphere_prefix=atmosphere_prefix,\
                                                      aerosol_prefix=aerosol_prefix,\
                                                      cloud_prefix=cloud_prefix,\
                                                      start_date=start_date,\
                                                      end_date=end_date,\
                                                      wavelengths_file=wavelengths_file,\
                                                      mask_file=mask_file,\
                                                      location='list',\
                                                      oasimlib_file=oasimlib_file,\
                                                      oasim_config_file=oasim_config_file,\
                                                      lats = np.array([lats[i]]),\
                                                      lons = np.array([lons[i]]),\
                                                      TheMask = TheMask
                                                      )
            indexes_.append(i)
            edout,esout = np.mean(oasim_output['edout'],axis=0)[:,0],np.mean(oasim_output['esout'],axis=0)[:,0]

            edout_.append(edout)
            esout_.append(esout)
            dates_.append(dates_end[i])
        print('rank:',rank,' reporting index',i,'date,',dates_start[i])
    edout_ = np.array(edout_).astype(np.float32)
    esout_ = np.array(esout_).astype(np.float32)

    edout_list = np.empty((len(dates_end),33))
    esout_list = np.empty((len(dates_end),33))
    
    if rank == 0:
        edout_list[0::nranks] = edout_
        esout_list[0::nranks] = esout_
        
    comm.Barrier()
    
    if rank != 0:
        comm.Send([edout_,MPI.FLOAT],dest=0,tag=0)
        comm.Send([esout_,MPI.FLOAT],dest=0,tag=1)
    else:
        if nranks!=1:
            for i in range(1,nranks):
                edout_recv = np.empty(edout_list[i::nranks].shape).astype(np.float32)
                comm.Recv([edout_recv,MPI.FLOAT],source=i,tag=0)
                edout_list[i::nranks] = edout_recv
                esout_recv = np.empty(esout_list[i::nranks].shape).astype(np.float32)
                comm.Recv([esout_recv,MPI.FLOAT],source=i,tag=1)
                esout_list[i::nranks] = esout_recv
        else:
            pass
        
    comm.Bcast(edout_list, root=0)
    comm.Bcast(esout_list, root=0)


    if rank == 0:
        if save == True:
            np.save(data_path + '/' + prefix + '_edout.npy',edout_list)
            np.save(data_path + '/' + prefix + '_esout.npy',esout_list)
            
    return edout_list,esout_list
                
    
def init_ncfile(len_lats,dates,lats,lons,chla_data,rrs_subsampled,rrs_std):
    wl = pd.read_csv('/g100_work/OGS23_PRACE_IT/csoto/DIIM/extern/OASIM_ATM/test/data/bin.txt', delim_whitespace=True, header=None).to_numpy()
    wl = np.mean(wl,1).astype(int)
    ncfile = nc.Dataset('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/oasim_output.nc','w')
    ncfile.createDimension('points',size=len_lats)
    ncfile.createDimension('wl',size = 33)
    ncfile.createDimension('wl_reduced',size=5)
    ncfile.createVariable('points','f4',('points'))
    ncfile.createVariable('wl','f4',('wl'))
    ncfile.createVariable('wl_reduced','f4',('wl_reduced'))
    ncfile.createVariable('edout','f4',('wl','points'))
    ncfile.createVariable('esout','f4',('wl','points'))
    ncfile.createVariable('date','f4',('points'))
    ncfile.createVariable('lat','f4',('points'))
    ncfile.createVariable('lon','f4',('points'))
    ncfile.createVariable('chla_in_situ','f4',('points'))
    ncfile.createVariable('chla_inversion','f4',('points'))
    ncfile.createVariable('rrs_satellite','f4',('wl_reduced','points'))
    ncfile.createVariable('rrs_std','f4',('wl_reduced','points'))
    ncfile['date'][:] = np.array([(date - datetime(year=1995,month=1,day=1)).days for date in dates.compressed()])
    ncfile['date'].units = 'days since January 01, 1995'
    ncfile['wl_reduced'][:] = np.array([412.5,443.5,490.0,510.0,555.0])
    ncfile['lat'][:] = lats.compressed()
    ncfile['lon'][:] = lons.compressed()
    ncfile['chla_in_situ'][:] = chla_data.compressed()
    ncfile['rrs_satellite'][:] = (rrs_subsampled.data[~lats.mask]).T
    ncfile['rrs_std'][:] = (rrs_std.data[~lats.mask]).T
    ncfile['wl'][:] = wl
    ncfile['points'][:] = np.arange(len_lats)
    
    ncfile.close()


def get_rrs_from_files(rrs_path,lats,lons,dates):
    len_all = len(dates)
    rrs_mask = np.ones(len_all)
    rrs_mask = (rrs_mask<0)

    rrs_labels = ['RRS412','RRS443','RRS490','RRS510','RRS555']
    rrs_subsampled = np.zeros((len_all,5))
    rrs_std = np.zeros((len_all,5))

    pixels_surrounding = 2

    for i in tqdm(range(len_all)):
        date_str = dates[i].strftime('%Y%m%d_cmems_obs-oc_med_bgc-reflectance_my_l3-multi-1km_P1D.nc')
        rrs_data_path = rrs_path + '/' + date_str
        if ~np.array(os.path.exists(rrs_data_path)):
            rrs_mask[i] = True
            continue
    
        rrs_data = nc.Dataset(rrs_data_path)
        
        lat_inf = np.argwhere( (rrs_data['lat'] < lats[i]) )[-1]
        lat_sup = np.argwhere( (rrs_data['lat'] > lats[i]) )[0]
        if lat_inf >= pixels_surrounding:
            lat_range_inf = lat_inf - pixels_surrounding
        else:
            lat_range_inf = 0
        if lat_sup < len(rrs_data['lat']) - pixels_surrounding:
            lat_range_sup = lat_sup + pixels_surrounding
        else:
            lat_range_sup = -1
    
        lon_inf = np.argwhere( (rrs_data['lon'] < lons[i]) )[-1]
        lon_sup = np.argwhere( (rrs_data['lon'] > lons[i]) )[0]
    
        if lon_inf >= pixels_surrounding:
            lon_range_inf = lon_inf - pixels_surrounding
        else:
            lon_range_inf = 0
        if lon_sup < len(rrs_data['lon']) - pixels_surrounding:
            lon_range_sup = lon_sup + pixels_surrounding
        else:
            lon_range_sup = -1

        for j,rrs_label in enumerate(rrs_labels):
            rrs_range = rrs_data[rrs_label][0,int(lat_range_inf):int(lat_range_sup),int(lon_range_inf):int(lon_range_sup)]
            if len(rrs_range.compressed())<((pixels_surrounding*2+1)**2)/2:
                rrs_mask[i] = True
        
            rrs_subsampled[i,j] = np.ma.mean(rrs_range)
            rrs_std[i,j] = np.ma.std(rrs_range)
            if (rrs_std[i,j]/rrs_subsampled[i,j]) > 0.15:
                rrs_mask[i] = True
    return rrs_subsampled,rrs_std,rrs_mask


if __name__ == '__main__':

    #Im using mpi to compute multiple points at the same time, each with different cores. 
    comm = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nranks = comm.size
    #in case the code try to be run in GPU, lets specify master as local host.
    torch.set_num_threads(1)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    #save_matches(save=True)
    ####test_data
    """
    N=100
    rrs_path = '/g100_work/OGS23_PRACE_IT/csoto/rrs_data/V11C/SAT/DAILY/CHECKED'
    x_data = np.load('/g100_work/OGS23_PRACE_IT/csoto/DIIM/settings/npy_data/x_data_all.npy')
    y_data = np.load('/g100_work/OGS23_PRACE_IT/csoto/DIIM/settings/npy_data/y_data_all.npy')
    chla_target = y_data[:N,0]
    rrs_ideal = x_data[:N,:5]
    edif_ideal = x_data[:N,5:10]
    edir_ideal = x_data[:N,10:15]
    par_ideal = x_data[:N,-5]
    zenith_ideal = x_data[:N,-6]
    dates = x_data[:,-1]
    lats = np.array([4.33666670E+01]*len(dates))[:N]
    lons = np.array([7.9]*len(dates))[:N]
    dates = np.array([datetime(year=2000,month=1,day=1,hour=11) + timedelta(days=date) for date in dates[:N] ])
    rrs_subsampled,rrs_std,rrs_mask = get_rrs_from_files(rrs_path,lats,lons,dates)
    lats = lats[~rrs_mask]
    lons = lons[~rrs_mask]
    dates = dates[~rrs_mask]
    chla_target = chla_target[~rrs_mask]
    rrs_subsampled = rrs_subsampled[~rrs_mask]
    rrs_ideal = rrs_ideal[~rrs_mask]
    edif_ideal = edif_ideal[~rrs_mask]
    edir_ideal = edir_ideal[~rrs_mask]
    par_ideal = par_ideal[~rrs_mask]
    zenith_ideal = zenith_ideal[~rrs_mask]
    dates_start = np.array([datetime(year=date.year,month=date.month,day=date.day,hour=7) for date in dates])
    dates_end = np.array([datetime(year=date.year,month=date.month,day=date.day,hour=16) for date in dates])
    edout,esout = generate_oasim_list(dates_start,dates_end,lats,lons,rank=rank,nranks=nranks,read=False,save=False)
    wavelengths_file = '/g100_work/OGS23_PRACE_IT/csoto/DIIM/extern/OASIM_ATM/test/data/bin.txt'
    wl = pd.read_csv(wavelengths_file, delim_whitespace=True, header=None).to_numpy()
    wl = np.mean(wl,1).astype(int)
    par = dm.PAR_calculator(wl,edout.T,esout.T,folder = '/g100_work/OGS23_PRACE_IT/csoto/diim_matches',save=False,read=False).T
    zenith = np.array([ dm.get_solar_position(dates[i],lats[i],lons[i],save=False,read=False) for i in range(len(dates)) ])
    dataset = load_data_tensor_list(rrs_subsampled,par,zenith,esout,edout,wl,dates,lons,lats)

    #slite difference in edif and edir (more light than the obtained...)
    #par is also different
    constant = rdm.read_constants(file1='/g100_work/OGS23_PRACE_IT/csoto/DIIM/settings/cte_lambda.csv',file2='/g100_work/OGS23_PRACE_IT/csoto/DIIM/settings/cte.csv')
    lr = 0.029853826189179603
    x_a = torch.zeros(3)
    s_a = torch.eye(3) * 4.9
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf
    batch_size = len_batch = len(dates)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = fm.Forward_Model(num_days=len_batch).to('cpu')
    chla_NN = dm.local_initial_conditions_nn(model,constant,next(iter(dataloader))).numpy()
    corr = np.corrcoef(np.exp(chla_NN[:,0]) ,chla_target)
    
    print(corr)

    sys.exit()
    """

    N=-1
    index=80

    #if rank == 0:
    #    dates,rrs_subsampled,lats,lons,chla_data,rrs_std,chla_data,rrs_mask = save_matches(save=True)
    #comm.Barrier()

    dates = np.load('dates.npy',allow_pickle = True)

    dates = np.array([date + timedelta(hours=11) for date in dates])
    rrs_subsampled = np.load('rrs_subsampled.npy')
    lats = np.load('lats.npy')
    lons = np.load('lons.npy')
    chla_data = np.load('chla_data.npy')

    rrs_std = np.load('rrs_std.npy')
    chla_mask = np.load('chla_mask.npy')
    rrs_mask = np.load('rrs_mask.npy')
        
    dates = np.ma.array(dates,mask = chla_mask)
    rrs_subsampled = np.ma.array(rrs_subsampled,mask = rrs_mask)
    lats = np.ma.array(lats,mask = chla_mask)
    lons = np.ma.array(lons,mask = chla_mask)
    chla_data = np.ma.array(chla_data,mask = chla_mask)
    rrs_std = np.ma.array(rrs_std,mask = rrs_mask)
    len_data = len(lats.compressed())

    dates_start = np.array([datetime(year=date.year,month=date.month,day=date.day,hour=7) for date in dates.compressed()])

    dates_end = np.array([datetime(year=date.year,month=date.month,day=date.day,hour=16) for date in dates.compressed()])

    #edout,esout = generate_oasim_list(dates_start[:],dates_end[:],lats.compressed()[:],lons.compressed()[:],rank=rank,nranks=nranks,read=False,save=True,data_path = '/g100_work/OGS23_PRACE_IT/csoto/diim_matches')
    edout,esout = generate_oasim_list(dates_start[:],dates_end[:],lats.compressed()[:],lons.compressed()[:],rank=rank,nranks=nranks,read=True,data_path = '/g100_work/OGS23_PRACE_IT/csoto/diim_matches')

    dates_mask = np.array([False if date > datetime(year=1999,day=1,month=1) else True for date in dates_start])

    
    lats = lats.compressed()
    lons = lons.compressed()
    
    #
    lats_mask = (lats < -999)
    lons_mask = (lons < -999)
    #lats_mask = (lats < 42.18) | (lats > 44.58) #proxi for liguria
    #lons_mask = (lons < 6.08) | (lons > 10.91)
    #
    lats = lats[~(dates_mask | lats_mask | lons_mask)][:N]
    lons = lons[~(dates_mask | lats_mask | lons_mask)][:N]
    
    edout = edout[~(dates_mask| lats_mask | lons_mask)][:N]
    esout = esout[~(dates_mask| lats_mask | lons_mask)][:N]
    dates = dates.compressed()[~(dates_mask| lats_mask | lons_mask)][:N]
        
    rrs_subsampled = rrs_subsampled.data[~chla_data.mask][~(dates_mask| lats_mask | lons_mask)][:N]
    rrs_std = rrs_std.data[~chla_data.mask][~(dates_mask| lats_mask | lons_mask)][:N]
    chla_data = chla_data.compressed()[~(dates_mask| lats_mask | lons_mask)][:N]
    

    wavelengths_file = '/g100_work/OGS23_PRACE_IT/csoto/DIIM/extern/OASIM_ATM/test/data/bin.txt'
    wl = pd.read_csv(wavelengths_file, delim_whitespace=True, header=None).to_numpy()
    wl = np.mean(wl,1).astype(int)
    par = dm.PAR_calculator(wl,edout.T,esout.T,folder = '/g100_work/OGS23_PRACE_IT/csoto/diim_matches',save=False,read=True,filename_date=False,filename='par.npy').T
    
    if rank == 0:
        if os.path.exists('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/zenith.npy'):
            zenith = np.load('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/zenith.npy')
        else:
            zenith = np.array([ dm.get_solar_position(dates[i],lats[i],lons[i],save=False,read=False) for i in range(len(dates)) ])
            np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/zenith.npy',zenith)
    comm.Barrier()
    if rank != 0:
        zenith = np.load('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/zenith.npy')
    
    
    #rrs_ = rrs_subsampled.copy()
    #rrs_subsampled[2,0] = rrs_[2,4]
    #rrs_subsampled[2,1] = rrs_[2,3]
    #rrs_subsampled[2,2] = rrs_[2,2]
    #rrs_subsampled[2,3] = rrs_[2,1]
    #rrs_subsampled[2,4] = rrs_[2,0]

    if os.path.exists('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/chla_log.npy'):
        chla_log = np.load('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/chla_log.npy')
        chla_NN = np.load('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/chla_NN.npy')
        RRS_log = np.load('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/RRS_log.npy')
    else:
        constant = rdm.read_constants(file1='/g100_work/OGS23_PRACE_IT/csoto/DIIM/settings/cte_lambda_dukiewicz/cte_lambda.csv',file2='/g100_work/OGS23_PRACE_IT/csoto/DIIM/settings/cte.csv')
        
        output = diim_list(rrs_subsampled[:],par[:],zenith[:],esout[:],edout[:],wl,dates[:],lons[:],lats[:],constant=constant,rank=rank,nranks=nranks,save = True)
        if rank == 0:
            datalist,chla_NN,chla_log,RRS_log = output
        comm.Barrier()
        chla_log = np.load('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/chla_log.npy')

    chla_data_inverted = np.exp(chla_log[:,0])

    corr = np.corrcoef(chla_data,chla_data_inverted)
    import matplotlib.pyplot as plt
    plt.plot(chla_data,chla_data_inverted)
    plt.show()

    
                                        
                         
    
    #######################################

    
    


    

