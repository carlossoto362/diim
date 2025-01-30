
from diim_map import set_mpi,inversion,create_map
from diimpy.read_data_module import read_constants
import torch
import os
import sys
import numpy as np

if __name__ == "__main__":

    if len(sys.argv)>=2:
        frack_files = int(sys.argv[1])
    else:
        frack_files = 1

    if len(sys.argv)==3:
        select_files = int(sys.argv[2]) - 1
    else:
        select_files = 0
    comm,rank,nranks = set_mpi('127.0.0.1','29500',1)
    
    scratch_path = '/g100_scratch/userexternal/csotolop/DIIM_output/diim_scratch'
    perturbation_factors_path = '/g100_work/OGS23_PRACE_IT/csoto/DIIM/settings/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy'
    constants_path = '/g100_work/OGS23_PRACE_IT/csoto/DIIM/settings'
    rrs_path = '/g100_work/OGS23_PRACE_IT/csoto/rrs_data/V11C/SAT/WEEKLY_24'
    oasim_path = '/g100_scratch/userexternal/csotolop/DIIM_output/OASIM_maps'
    PAR_path = '/g100_scratch/userexternal/csotolop/DIIM_output/PAR'
    zenith_path = '/g100_scratch/userexternal/csotolop/DIIM_output/zenith'
    output_path = '/g100_scratch/userexternal/csotolop/DIIM_output/diim_maps'
    
    perturbation_factors = torch.tensor(np.load(perturbation_factors_path))[-100:].mean(axis=0).to(torch.float32)
    my_device = 'cpu'
    constant = read_constants(file1=constants_path+'/cte_lambda.csv',file2=constants_path+'/cte.csv',my_device = my_device)
    

    
    rrs_files = os.listdir(rrs_path)

    init = int(len(rrs_files)/frack_files)*select_files
    if frack_files == select_files + 1:
        end = len(rrs_files)
    else:
        end = int(len(rrs_files)/frack_files)*(select_files+1)
    print(frack_files,select_files)
    rrs_files_ = rrs_files[init:end]

    if rank == 0:
        print('creating {} maps'.format(len(rrs_files_)))
        import time
    
    for file_ in rrs_files_:
        
        if rank == 0:
            time_init = time.time()

        date_str = file_.split('_')[0]
        dateformat = '%Y%m%d'
        oasim_data_path = oasim_path + '/oasim_map_' + date_str + '.nc'
        rrs_data_path = rrs_path + '/' + date_str + '_cmems_obs-oc_med_bgc-reflectance_my_l3-multi-1km_P1D.nc'
        output_data_path = output_path + '/diim_map_' + date_str + '.nc'

        if ~os.path.isfile(oasim_data_path): continue
        dataset = inversion(dateformat=dateformat,scratch_path=scratch_path,oasim_data_path=oasim_data_path,rrs_data_path=rrs_data_path,PAR_path=PAR_path,zenith_path=zenith_path,date_str=date_str,rank=rank,nranks=nranks,comm=comm,my_device=my_device,perturbation_factors = perturbation_factors,constant = constant)    
    
        if rank == 0:
            create_map(scratch_path,dataset,output_path = output_data_path,date_str = date_str)
            print(output_data_path + 'created in {} seconds.'.format(time.time() - time_init))

