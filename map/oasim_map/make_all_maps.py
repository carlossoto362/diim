from oasim_map import getting_oasim_output,creat_oasim_netcdf
from datetime import datetime, timedelta
from mpi4py import MPI
import os

if __name__ == '__main__':

    #Im using mpi to compute multiple points at the same time, each with different cores. 
    comm = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nranks = comm.size
    #in case the code try to be run in GPU, lets specify master as local host. 
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    day_0 = datetime(year=2000,month=1,day=1)
    last_day = datetime(year=2020,month=12,day=31)
    num_days = (last_day - day_0).days

    dateformat = '%Y%m%d-%H:%M:%S'
    aerosol_datadir = '/g100_work/OGS_devC/NECCTON/OPTICS'
    cloud_datadir = '/g100_work/OGS_devC/NECCTON/OPTICS'
    atmosphere_prefix = 'atm'
    aerosol_prefix = 'aero'
    cloud_prefix = 'climatm'
    mask_file = '/g100_work/OGS_devC/V9C/RUNS_SETUP/PREPROC/MASK/meshmask.nc'
    location = 'map'
    wavelengths_file = '/g100_work/OGS23_PRACE_IT/csoto/DIIM/extern/OASIM_ATM/test/data/bin.txt'
    oasimlib_file = '/g100_work/OGS23_PRACE_IT/csoto/DIIM/map/oasim_map/liboasim-py.so'
    oasim_config_file = '/g100_work/OGS23_PRACE_IT/csoto/DIIM/map/oasim_map/oasim_config.yaml'
    output_prefix = 'oasim_med'
    output_datadir = '/g100_scratch/userexternal/csotolop/OASIM_maps'
    
    for i in range(rank,num_days,nranks):
        oasim_output = getting_oasim_output(dateformat=dateformat,\
                                            atmosphere_datadir=(day_0 + timedelta(days=i)).strftime('/g100_work/OGS_devC/NECCTON/OPTICS/%Y/%m'),\
                                            aerosol_datadir=aerosol_datadir,\
                                            cloud_datadir=cloud_datadir,\
                                            atmosphere_prefix=atmosphere_prefix,\
                                            aerosol_prefix=aerosol_prefix,\
                                            cloud_prefix=cloud_prefix,\
                                            start_date=(day_0 + timedelta(days=i)).strftime('%Y%m%d-07:00:00'),\
                                            end_date=(day_0 + timedelta(days=i)).strftime('%Y%m%d-16:00:00'),\
                                            wavelengths_file=wavelengths_file,\
                                            mask_file=mask_file,\
                                            location=location,\
                                            oasimlib_file=oasimlib_file,\
                                            oasim_config_file=oasim_config_file
                                            )
        
            
        output_name = 'oasim_map_' + oasim_output['times'][0].strftime('%Y%m%d.nc')
        creat_oasim_netcdf(times=oasim_output['times'],
                           wl=oasim_output['wl'],
                           lat=oasim_output['lat'],
                           lon=oasim_output['lon'],
                           sp=oasim_output['sp'],
                           msl=oasim_output['msl'],
                           ws10=oasim_output['ws10'],
                           tco3=oasim_output['tco3'],
                           t2m=oasim_output['t2m'],
                           d2m=oasim_output['d2m'],
                           tcc=oasim_output['tcc'],
                           tclw=oasim_output['tclw'],
                           cdrem=oasim_output['cdrem'],
                           taua=oasim_output['taua'],
                           asymp=oasim_output['asymp'],
                           ssalb=oasim_output['ssalb'],
                           edout=oasim_output['edout'],
                           esout=oasim_output['esout'],
                           output_name=output_name,
                           output_datadir=output_datadir)

        print('rank {} reporting the creation of map'.format(rank),output_name)
        del oasim_output
            


