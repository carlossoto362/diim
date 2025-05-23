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
        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/dates.npy',dates.data)
        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/rrs_subsampled.npy',rrs_subsampled.data)
        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/lats.npy',lats.data)
        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/lons.npy',lons.data)
        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/chla_data.npy',chla_data.data)

        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/rrs_std.npy',rrs_std.data)
        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/chla_mask.npy',chla_data.mask)
        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/rrs_mask.npy',rrs_subsampled.mask)
        
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
        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/chla_NN.npy',chla_NN_rec)
        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/chla_log.npy',chla_log_rec)
        np.save('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/RRS_log.npy',RRS_log_rec)
        return dataset,chla_NN_rec,chla_log_rec,RRS_log_rec






def generate_oasim_list(dates_start,dates_end,lats,lons,rank=0,nranks=1,save=False,read=False,data_path = './',prefix=''):
    if read == True:
        edout = np.load(data_path + '/' + prefix + '_edout.npy')
        esout = np.load(data_path + '/' + prefix + '_esout.npy') 
        return edout,esout
    
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


from shapely.geometry import Point, Polygon

    # Dictionary of sub-basin polygons (approximate!)
subbasin_polygons = {
        "Western Mediterranean": Polygon([
            (-5.5, 35.0), (9.0, 35.0), (9.0, 43.5), (-5.5, 43.5)
        ]),
        "Adriatic Sea": Polygon([
            (12.0, 39.5), (20.0, 39.5), (20.0, 45.8), (12.0, 45.8)
        ]),
        "Ionian Sea": Polygon([
            (14.0, 34.0), (22.0, 34.0), (22.0, 40.0), (14.0, 40.0)
        ]),
        "Aegean Sea": Polygon([
            (22.0, 35.0), (28.0, 35.0), (28.0, 41.0), (22.0, 41.0)
        ]),
        "Levantine Basin": Polygon([
            (28.0, 30.0), (36.0, 30.0), (33.0, 33.5), (28.0, 36.5)
        ]),
        "Tyrrhenian Sea": Polygon([
            (8.5, 38.0), (15.5, 38.0), (15.5, 42.5), (8.5, 42.5)
        ]),
        "Alboran Sea": Polygon([
            (-5.5, 35.0), (0.5, 35.0), (0.5, 37.5), (-5.5, 37.5)
        ]),
        "Balearic Sea": Polygon([
            (0.5, 37.5), (6.5, 37.5), (6.5, 41.5), (0.5, 41.5)
        ]),
        "Central Mediterranean": Polygon([
            (10.0, 33.0), (22.0, 33.0), (22.0, 38.0), (10.0, 38.0)
        ]),
        "BOUSSOLE": Polygon([
            (6.5, 42.0), (9.5, 42.0), (9.5, 45), (6.5, 45)
        ])
    }

def get_subbasin_mask(lats, lons, subbasin_name):
    """
    Returns a boolean mask for points in the specified Mediterranean sub-basin using polygon boundaries.

    Parameters:
        lats (np.ndarray): Array of latitudes (N,)
        lons (np.ndarray): Array of longitudes (N,)
        subbasin_name (str): Name of the sub-basin

    Returns:
        np.ndarray: Boolean mask array of shape (N,)
    """

    if (subbasin_name not in subbasin_polygons) :
        if type(subbasin_name) == int:
            names = ["Western Mediterranean","Adriatic Sea","Ionian Sea","Aegean Sea","Levantine Basin","Tyrrhenian Sea","Alboran Sea","Balearic Sea","Central Mediterranean","BOUSSOLE"]
            subbasin_name = names[subbasin_name]
        else:
            raise ValueError(f"Unknown sub-basin: {subbasin_name}")


    polygon = subbasin_polygons[subbasin_name]
    
    # Normalize longitudes to [-180, 180]
    lons = ((lons + 180) % 360) - 180

    # Generate mask by checking if each point is inside the polygon
    mask = np.array([polygon.contains(Point(lon, lat)) for lat, lon in zip(lats, lons)])

    return mask

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon as MplPolygon

def plot_mediterranean_polygon_with_points(subbasin_name, lats=None, lons=None,ax=None,batimetry=None):
    """
    Plots a Mediterranean map using Basemap, with a sub-basin polygon and optional data points.

    Parameters:
        subbasin_name (str): Name of the sub-basin to highlight.
        lats (np.ndarray): Optional array of latitudes.
        lons (np.ndarray): Optional array of longitudes.
    """
    # Define polygon boundaries (approximated)
    subbasin_polygons = {
        "Western Mediterranean": [(-5.5, 35.0), (9.0, 35.0), (9.0, 43.5), (-5.5, 43.5)],
        "Adriatic Sea": [(12.0, 39.5), (20.0, 39.5), (20.0, 45.8), (12.0, 45.8)],
        "Ionian Sea": [(14.0, 34.0), (22.0, 34.0), (22.0, 40.0), (14.0, 40.0)],
        "Aegean Sea": [(22.0, 35.0), (28.0, 35.0), (28.0, 41.0), (22.0, 41.0)],
        "Levantine Basin": [(28.0, 30.0), (33.0, 30.0), (33.0, 36.5), (28.0, 36.5)],
        "Tyrrhenian Sea": [(8.5, 38.0), (15.5, 38.0), (15.5, 42.5), (8.5, 42.5)],
        "Alboran Sea": [(-5.5, 35.0), (0.5, 35.0), (0.5, 37.5), (-5.5, 37.5)],
        "Balearic Sea": [(0.5, 37.5), (6.5, 37.5), (6.5, 41.5), (0.5, 41.5)],
        "Central Mediterranean": [(10.0, 33.0), (22.0, 33.0), (22.0, 38.0), (10.0, 38.0)],
        "BOUSSOLE": [ (6.5, 42.0), (9.5, 42.0), (9.5, 45), (6.5, 45)]
    }
    if (subbasin_name not in subbasin_polygons) :
        if type(subbasin_name) == int:
            names = ["Western Mediterranean","Adriatic Sea","Ionian Sea","Aegean Sea","Levantine Basin","Tyrrhenian Sea","Alboran Sea","Balearic Sea","Central Mediterranean","BOUSSOLE"]
            subbasin_name = names[subbasin_name]
        else:
            raise ValueError(f"Unknown sub-basin: {subbasin_name}")
    
    poly_coords = subbasin_polygons[subbasin_name]

    # Setup map
    m = Basemap(
        projection='cyl',
        llcrnrlon=poly_coords[0][0]-1,   
        urcrnrlon=poly_coords[1][0]+2,
        llcrnrlat=poly_coords[0][1]-2,
        urcrnrlat=poly_coords[2][1]+1,
        resolution='i',
        ax=ax
    )

    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawparallels(np.arange(30, 47, 2), labels=[1, 0, 0, 0],fontsize=20)
    m.drawmeridians(np.arange(-10, 39, 5), labels=[0, 0, 0, 1],fontsize=20)
    
    

    # Draw polygon
    polygon_pts = [(lon, lat) for lon, lat in poly_coords]
    poly = MplPolygon(polygon_pts, closed=True, edgecolor='red', facecolor='red', alpha=0.15, linewidth=2)
    ax.add_patch(poly)
    ax.plot(*zip(*polygon_pts), color='red')

    # Plot points if provided
    if lats is not None and lons is not None:
        lons = np.asarray(lons)
        lats = np.asarray(lats)
        x, y = m(lons, lats)
        m.scatter(x, y, s=70, marker = 'x' ,c='black', label='Data points', zorder=5)

    if type(batimetry) != type(None):

        def fmt(x):
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s} m" if plt.rcParams["text.usetex"] else f"{s} m"
        CS = m.contour(batimetry['lon'][:], batimetry['lat'][:], batimetry['depth'][:], levels=4, linewidths=0.5, colors='k',ls='--')
        
        ax.clabel(CS, CS.levels, fmt=fmt, fontsize=10)
        
    ax.tick_params(axis="y", labelsize=40)
    ax.tick_params(axis="x", labelsize=40)
    ax.text(poly_coords[0][0]-1,poly_coords[2][1]+1.05,'(A)',fontsize=25)
    #ax.set_title(f"{subbasin_name}")
    #ax.legend()
    ax.grid(True)



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
    depths = np.load('depths.npy')

    rrs_std = np.load('rrs_std.npy')
    chla_mask = np.load('chla_mask.npy')
    rrs_mask = np.load('rrs_mask.npy')
    print('data loaded')

    #rrs_mask = rrs_mask | np.array([(chla_data<0.1)]*5).T
        
    dates = np.ma.array(dates,mask = chla_mask)
    rrs_subsampled = np.ma.array(rrs_subsampled,mask = rrs_mask)
    lats = np.ma.array(lats,mask = chla_mask)
    lons = np.ma.array(lons,mask = chla_mask)
    chla_data = np.ma.array(chla_data,mask = chla_mask)
    rrs_std = np.ma.array(rrs_std,mask = rrs_mask)
    depths = np.ma.array(depths,mask = chla_mask)
    len_data = len(lats.compressed())

    dates_start = np.array([datetime(year=date.year,month=date.month,day=date.day,hour=7) for date in dates.compressed()])

    dates_end = np.array([datetime(year=date.year,month=date.month,day=date.day,hour=16) for date in dates.compressed()])

    #edout,esout = generate_oasim_list(dates_start[:],dates_end[:],lats.compressed()[:],lons.compressed()[:],rank=rank,nranks=nranks,read=False,save=True,data_path = '/g100_work/OGS23_PRACE_IT/csoto/diim_matches')
    edout,esout = generate_oasim_list(dates_start[:],dates_end[:],lats.compressed()[:],lons.compressed()[:],rank=rank,nranks=nranks,read=True,data_path = './')

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
    depths = depths.compressed()[~(dates_mask| lats_mask | lons_mask)][:N]
    
    

    wavelengths_file = 'bin.txt'
    wl = pd.read_csv(wavelengths_file, delim_whitespace=True, header=None).to_numpy()
    wl = np.mean(wl,1).astype(int)
    par = dm.PAR_calculator(wl,edout.T,esout.T,folder = './',save=False,read=True,filename_date=False,filename='par.npy').T
    
    if rank == 0:
        if os.path.exists('zenith.npy'):
            zenith = np.load('zenith.npy')
        else:
            zenith = np.array([ dm.get_solar_position(dates[i],lats[i],lons[i],save=False,read=False) for i in range(len(dates)) ])
            np.save('zenith.npy',zenith)
    comm.Barrier()
    if rank != 0:
        zenith = np.load('zenith.npy')
    
    
    #rrs_ = rrs_subsampled.copy()
    #rrs_subsampled[2,0] = rrs_[2,4]
    #rrs_subsampled[2,1] = rrs_[2,3]
    #rrs_subsampled[2,2] = rrs_[2,2]
    #rrs_subsampled[2,3] = rrs_[2,1]
    #rrs_subsampled[2,4] = rrs_[2,0]

    if os.path.exists('chla_log.npy'):
        chla_log = np.load('chla_log.npy')
        chla_NN = np.load('chla_NN.npy')
        RRS_log = np.load('RRS_log.npy')
    else:
        constant = rdm.read_constants(file1='g100_work/OGS23_PRACE_IT/csoto/DIIM/settings/cte_lambda_dukiewicz/cte_lambda.csv',file2='/g100_work/OGS23_PRACE_IT/csoto/DIIM/settings/cte.csv')
        
        output = diim_list(rrs_subsampled[:],par[:],zenith[:],esout[:],edout[:],wl,dates[:],lons[:],lats[:],constant=constant,rank=rank,nranks=nranks,save = True)
        if rank == 0:
            datalist,chla_NN,chla_log,RRS_log = output
        comm.Barrier()
        chla_log = np.load('/g100_work/OGS23_PRACE_IT/csoto/diim_matches/chla_log.npy')

    chla_data_inverted = np.exp(chla_NN[:,0])



    
    ###########################################################################################################
    ###########################################flag for batimetry depth and season#############################
    ###########################################################################################################
    if os.path.exists('batimetry.npy'):
        batimetry = np.load('batimetry.npy')
    else:
        batimetry = nc.Dataset('depth.nc')

        def depth_lan_lot(lat,lon):
            lat_inf = np.argwhere( (batimetry['lat'][:,0] <= lat) )[-1]
            lat_sup = np.argwhere( (batimetry['lat'][:,0] > lat) )[0]

            lon_inf = np.argwhere( (batimetry['lon'][0,:] <= lon) )[-1]
            lon_sup = np.argwhere( (batimetry['lon'][0,:] > lon) )[0]

            mean_depth = (batimetry['depth'][lat_inf,lon_inf] + batimetry['depth'][lat_inf,lon_sup]+\
                          batimetry['depth'][lat_sup,lon_inf] + batimetry['depth'][lat_sup,lon_sup])/4
            return mean_depth[0,0]

        batimetry_npy = []
        for lat_i, lon_i in zip(lats,lons):
            batimetry_npy.append(depth_lan_lot(lat_i,lon_i))
        batimetry_npy = np.array(batimetry_npy)
        np.save('batimetry.npy',batimetry_npy)
        batimetry = batimetry_npy


    batimetry_mask = (batimetry < 400)
    depth_mask = (depths > 10)
    chla_mask = (chla_data<0)
    
    chla_data = chla_data[(~batimetry_mask) & (~depth_mask) & (~chla_mask)]
    depths = depths[(~batimetry_mask) & (~depth_mask)& (~chla_mask)]
    dates = dates[(~batimetry_mask) & (~depth_mask)& (~chla_mask)]
    chla_data_inverted = chla_data_inverted[(~batimetry_mask) & (~depth_mask)& (~chla_mask)]
    batimetry = batimetry[(~batimetry_mask) & (~depth_mask)& (~chla_mask)]
    rrs_subsampled = rrs_subsampled[(~batimetry_mask) & (~depth_mask)& (~chla_mask)]
    lats = lats[(~batimetry_mask) & (~depth_mask)& (~chla_mask)]
    lons = lons[(~batimetry_mask) & (~depth_mask)& (~chla_mask)]

    ##########SeaWiFS maximum with ratio#########
    max_rrs = np.array([np.max([rrs1,rrs2,rrs3]) for rrs1,rrs2,rrs3 in zip(rrs_subsampled[:,1],rrs_subsampled[:,2],rrs_subsampled[:,3])])
    X = np.log(max_rrs/rrs_subsampled[:,4])/np.log(10)
    X_2 = X**2
    X_3 = X**3
    X_4 = X**4
    
    #from sklearn import linear_model
    
    X = np.array([X,X_2,X_3,X_4]).T
    y = np.log(chla_data)/np.log(10)
    #regr = linear_model.LinearRegression()
    #regr.fit(X[~train_mask], y[~train_mask])
    
    #chla_data_inverted = chla_seaWiFS
    def rmse(x1,x2):
        return np.sqrt( np.mean((x1 - x2)**2) )        

    #chla_seaWiFS = 10**(regr.predict(X[(~test_mask)&(~chla_mask)]))
    chla_seaWiFS = 10**(0.327 - 2.994*X[:,0] + 2.722*X[:,1] -1.226 * X[:,2] -0.568 * X[:,3])
    
    spring_mask = ~np.array([(date.month in [3,4,5]) for date in dates ])
    summer_mask = ~np.array([(date.month in [6,7,8]) for date in dates ])
    autum_mask = ~np.array([(date.month in [9,10,11]) for date in dates ])
    winter_mask = ~np.array([(date.month in [12,1,2]) for date in dates ])
    
    for subbasin_name in range(9,10):
        subbasin_mask =  ~(get_subbasin_mask(lats, lons, subbasin_name)) 
    #subbasin_name = 1
        #chla_mask = ((chla_data>2.5) & (chla_data_inverted<0.75)) | ~(get_subbasin_mask(lats, lons, subbasin_name))
        #corr = np.corrcoef(chla_data,chla_data_inverted)
        import matplotlib.pyplot as plt
        #print('spring')
        #print('rrs_seaWiFS',rmse(chla_seaWiFS[~spring_mask],chla_data[~spring_mask]))
        #print('rrs_inverted',rmse(chla_data_inverted[~spring_mask],chla_data[~spring_mask]))
        #print('summer')
        #print('rrs_seaWiFS',rmse(chla_seaWiFS[~summer_mask],chla_data[~summer_mask]))
        #print('rrs_inverted',rmse(chla_data_inverted[~summer_mask],chla_data[~summer_mask]))
        #print('winter')
        #print('rrs_seaWiFS',rmse(chla_seaWiFS[~winter_mask],chla_data[~winter_mask]))
        #print('rrs_inverted',rmse(chla_data_inverted[~winter_mask],chla_data[~winter_mask]))
        #print('autum')
        #print('rrs_seaWiFS',rmse(chla_seaWiFS[~autum_mask],chla_data[~autum_mask]))
        #print('rrs_inverted',rmse(chla_data_inverted[~autum_mask],chla_data[~autum_mask]))
        """
        plt.close('all')
        fig = plt.figure(figsize=(30,8))
        
        ax1 = plt.subplot2grid((2, 6), (0, 0),rowspan=2,colspan=2)
        
        ax2 = plt.subplot2grid((2, 6), (0, 2))
        ax3 = plt.subplot2grid((2, 6), (0, 3))
        ax4 = plt.subplot2grid((2, 6), (0, 4))
        ax5 = plt.subplot2grid((2, 6), (0, 5))
        axs1 = [ax2,ax3,ax4,ax5]
        
        ax6 = plt.subplot2grid((2, 6), (1, 2))
        ax7 = plt.subplot2grid((2, 6), (1, 3))
        ax8 = plt.subplot2grid((2, 6), (1, 4))
        ax9 = plt.subplot2grid((2, 6), (1, 5))
        axs2 = [ax6,ax7,ax8,ax9]
        
        
        plot_mediterranean_polygon_with_points(subbasin_name,lats[~subbasin_mask],lons[~subbasin_mask],ax=ax1)
        
        axs = axs1
        
        axs[0].scatter(chla_data[(~spring_mask) & (~subbasin_mask)],chla_data_inverted[(~spring_mask) & (~subbasin_mask)],label='Spring data, corr: {:.3f}'.format(np.corrcoef(chla_data[(~spring_mask) & (~subbasin_mask)],chla_data_inverted[(~spring_mask) & (~subbasin_mask)])[0,1]),color='purple',alpha=0.4,marker='o')
        
        axs[1].scatter(chla_data[(~summer_mask) & (~subbasin_mask)],chla_data_inverted[(~summer_mask) & (~subbasin_mask)],label='Summer data, corr: {:.3f}'.format(np.corrcoef(chla_data[(~summer_mask) & (~subbasin_mask)],chla_data_inverted[(~summer_mask) & (~subbasin_mask)])[0,1]),color='orange',alpha=0.4,marker = 'x')
        
        axs[2].scatter(chla_data[(~autum_mask) & (~subbasin_mask)],chla_data_inverted[(~autum_mask) & (~subbasin_mask)],label='Autum data, corr: {:.3f}'.format(np.corrcoef(chla_data[(~autum_mask) & (~subbasin_mask)],chla_data_inverted[(~autum_mask) & (~subbasin_mask)])[0,1]),color='red',alpha=0.4,marker='1')
        
        axs[3].scatter(chla_data[(~winter_mask) & (~subbasin_mask)],chla_data_inverted[(~winter_mask) & (~subbasin_mask)],label='Winter data, corr: {:.3f}'.format(np.corrcoef(chla_data[(~winter_mask) & (~subbasin_mask)],chla_data_inverted[(~winter_mask) & (~subbasin_mask)])[0,1]),color='blue',alpha=0.4,marker='+')
        
        for ax in axs:
            ax.set_xlabel('chlorophyll observations $[mgm^{-1}]$')
            ax.set_ylabel('chlorophyll inverted $[mgm^{-1}]$')
            ax.plot(np.linspace(0,3,20),np.linspace(0,3,20),'--',color='black',label='x=y')
            ax.legend()
        axs[0].set_xlim(0,3)
        axs[0].set_ylim(0,3)
        
        axs[1].set_xlim(0,1)
        axs[1].set_ylim(0,1)
        
        axs[2].set_xlim(0,1)
        axs[2].set_ylim(0,1)
        
        axs[3].set_xlim(0,1.5)
        axs[3].set_ylim(0,1.5)
            
        axs = axs2
            
        axs[0].scatter(chla_data[(~spring_mask) & (~subbasin_mask)],chla_seaWiFS[(~spring_mask) & (~subbasin_mask)],label='Spring data, corr: {:.3f}'.format(np.corrcoef(chla_data[(~spring_mask) & (~subbasin_mask)],chla_seaWiFS[(~spring_mask) & (~subbasin_mask)])[0,1]),color='purple',alpha=0.4,marker='o')
        
        axs[1].scatter(chla_data[(~summer_mask) & (~subbasin_mask)],chla_seaWiFS[(~summer_mask) & (~subbasin_mask)],label='Summer data, corr: {:.3f}'.format(np.corrcoef(chla_data[(~summer_mask) & (~subbasin_mask)],chla_seaWiFS[(~summer_mask) & (~subbasin_mask)])[0,1]),color='orange',alpha=0.4,marker = 'x')
        
        axs[2].scatter(chla_data[(~autum_mask) & (~subbasin_mask)],chla_seaWiFS[(~autum_mask) & (~subbasin_mask)],label='Autum data, corr: {:.3f}'.format(np.corrcoef(chla_data[(~autum_mask) & (~subbasin_mask)],chla_seaWiFS[(~autum_mask) & (~subbasin_mask)])[0,1]),color='red',alpha=0.4,marker='1')
        
        axs[3].scatter(chla_data[(~winter_mask) & (~subbasin_mask)],chla_seaWiFS[(~winter_mask) & (~subbasin_mask)],label='Winter data, corr: {:.3f}'.format(np.corrcoef(chla_data[(~winter_mask) & (~subbasin_mask)],chla_seaWiFS[(~winter_mask) & (~subbasin_mask)])[0,1]),color='blue',alpha=0.4,marker='+')
        
        for ax in axs:
            ax.set_xlabel('chlorophyll observations $[mgm^{-1}]$')
            ax.set_ylabel('chlorophyll seaWiFS $[mgm^{-1}]$')
            ax.plot(np.linspace(0,3,20),np.linspace(0,3,20),'--',color='black',label='x=y')
            ax.legend(fontsize=20)
        axs[0].set_xlim(0,3)
        axs[0].set_ylim(0,3)

        axs[1].set_xlim(0,1)
        axs[1].set_ylim(0,1)
        
        axs[2].set_xlim(0,1)
        axs[2].set_ylim(0,1)

        axs[3].set_xlim(0,1.5)
        axs[3].set_ylim(0,1.5)
        plt.tight_layout()
        plt.savefig('/home/carlos/Documents/TriesteUniversity/diim_matches/'+list(subbasin_polygons.keys())[subbasin_name]+'.png')
    """
        
        plt.close('all')
        fig = plt.figure(figsize=(16,8))
        
        ax1 = plt.subplot2grid((2, 2), (0, 0),rowspan=2,colspan=1)
        
        ax2 = plt.subplot2grid((2, 2), (0, 1))

        axs1 = [ax2]
        
        ax6 = plt.subplot2grid((2, 2), (1, 1))

        axs2 = [ax6]
        
        batimetry_nc = nc.Dataset('depth.nc')
        plot_mediterranean_polygon_with_points(subbasin_name,lats[~subbasin_mask],lons[~subbasin_mask],ax=ax1,batimetry=batimetry_nc)
        
        axs = axs1
        
        axs[0].scatter(chla_data[ (~subbasin_mask)],chla_data_inverted[ (~subbasin_mask)],label='Spring data, corr: {:.3f}'.format(np.corrcoef(chla_data[ (~subbasin_mask)],chla_data_inverted[ (~subbasin_mask)])[0,1]),color='#004D40',alpha=0.4,marker='o')
        
        for ax in axs:
            #ax.set_xlabel('chlorophyll observations $[mgm^{-1}]$',fontsize=20)
            ax.set_ylabel('chlorophyll inverted $[mgm^{-1}]$',fontsize=15)
            ax.plot(np.linspace(0,3,20),np.linspace(0,3,20),'--',color='black',label='x=y')
            ax.legend(fontsize=20)
            ax.tick_params(axis="x", labelsize=20)
            ax.tick_params(axis="y", labelsize=20)
        axs[0].set_xlim(0,3)
        axs[0].set_xticks([])
        axs[0].set_ylim(0,3)
        axs[0].text(-0.6,3.1,'(B)',fontsize=25)
            
        axs = axs2
            
        axs[0].scatter(chla_data[(~subbasin_mask)],chla_seaWiFS[ (~subbasin_mask)],label='Spring data, corr: {:.3f}'.format(np.corrcoef(chla_data[ (~subbasin_mask)],chla_seaWiFS[ (~subbasin_mask)])[0,1]),color='#1E88E5',alpha=0.4,marker='o')
               
        for ax in axs:
            ax.set_xlabel('chlorophyll observations $[mgm^{-1}]$',fontsize=20)
            ax.set_ylabel('chlorophyll MedOC4.2020 $[mgm^{-1}]$',fontsize=15)
            ax.plot(np.linspace(0,3,20),np.linspace(0,3,20),'--',color='black',label='x=y')
            ax.legend(fontsize=20)
            ax.tick_params(axis="x", labelsize=20)
            ax.tick_params(axis="y", labelsize=20)
        axs[0].set_xlim(0,3)
        axs[0].set_ylim(0,3)
        axs[0].text(-0.6,3.1,'(C)',fontsize=25)

        plt.tight_layout()
        plt.savefig(list(subbasin_polygons.keys())[subbasin_name]+'.pdf')

    
    
                                        
                         
    
    #######################################

    
    


    

