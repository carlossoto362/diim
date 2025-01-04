import argparse
import sys
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,random_split
from torch import nn
from tqdm import tqdm
from diimpy import read_data_module as rdm
from diimpy import Forward_module as fm
from diimpy import bayesian_inversion as bayes
import matplotlib.pyplot as plt
import scipy
from diimpy import sensitivity_analysis_and_mcmc_runs as mcmc
from diimpy import plot_data_lognormal as pdl
from datetime import datetime
from diimpy import CVAE_model_part_one as cvae_one
from diimpy import CVAE_model_part_two as cvae_two

if 'DIIM_PATH' in os.environ:
    HOME_PATH = MODEL_HOME = os.environ["DIIM_PATH"]
else:
    
    print("Missing local variable DIIM_PATH. \nPlease add it with '$:export DIIM_PATH=path/to/diim'.")
    sys.exit()

def argument():
    parser = argparse.ArgumentParser(description = '''
    Script to reproduce the results in DOI: https://doi.org/10.5194/gmd-2024-174.
    ''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument( '-a',
                         action='store_true',
                         help='Interactive script, use it to have a gided tour on how to reproduce the results.'
    )
    group.add_argument('-b',
                        action='store_true',
                        help='Only bayesian part'
    )
    group.add_argument('-m',
                       action='store_true',
                       help='Only sensitivity analysis and mcmc part'
    )
    group.add_argument('-n',
                       action='store_true',
                       help='Only the Neural Network part'
    )

                                                 
    return parser.parse_args()

def iprint(message,delay = 0.02):
    """Prints a message letter by letter, with a given delay between letters."""
    print("Interactive cat: - ",end='')
    
    for letter in message:
        sys.stdout.write(letter)  # Print letter without newline
        sys.stdout.flush()         # Ensure it appears immediately
        time.sleep(delay)          # Wait before printing the next letter
    print()  # Newline after the message is printed
    print('')
    
def cprint(message,output=None):
    print('>>>',end='')
    print(message)
    if output:
        print(output)
        
def iflag(flag=None):
    limit_tr = 0
    while limit_tr < 3:
        flag = input('User - ')
        flag = ("".join(flag.strip())).lower()
        if flag in ['yes','no']:
            return flag
        else:
            iprint('Please wirte yes or no.')
        limit_tr += 1
    sys.exit()
                       
        

def bayesian_part():
        
    iprint("""The first approach we tested was the Bayesian approach. As described in the README file (please read it) and explained in the test.py file (also read it—it’s meant to be read, not just run), the scripts needed to reproduce this part are in the diimpy folder and include:

    1) read_data_module.py: Used to read the data and transform the input into Torch tensors.
    2) Forward_module.py: Contains the forward computations, including all the formulas needed to compute the theoretical Remote Sensing Reflectance as a function of the OASIM model inputs, the optical constituents, and the empirical parameters from the model.
    3) bayesian_inversion.py: This module includes the scripts needed to perform the inversion (standalone) or in combination with alternate minimization (by optimizing the parameters simultaneously).""",delay = 0.002)

    cprint('from diimpy import read_data_module as rdm')
    cprint('from diimpy import Forward_module as fm')
    cprint('from diimpy iport bayesian_inversion as bayes')
    print('')

    iprint('To recreate the paper, we first run alternate minimization with an approximately uniform prior (alpha >> 1).')
    iprint('Would you like to run this part? (yes/no)')
    flag=iflag()
    if flag == 'yes':
        iprint('This part saves the history of perturbation_factors_used in DIIM_PATH + "/settings/reproduce/perturbation_factors/perturbation_factors_history_new.pt"')
        cprint("track_parameters(data_path = MODEL_HOME + '/settings/npy_data',output_path = MODEL_HOME + '/settings/reproduce/perturbation_factors',iterations=1000,save=True )")
        bayes.track_parameters(data_path = MODEL_HOME + '/settings/npy_data',output_path = MODEL_HOME + '/settings/reproduce/perturbation_factors',iterations=1000,save=True )
            
    print('')

            
    iprint('With these perturbation factors, we tuned the prior parameter alpha (see the paper, Appendix B). This step is necessary because, for each day, the inversion is performed using only five wavelengths. With such limited data, the prior significantly influences the uncertainty. Our approach to tuning alpha is similar to Bayesian model specification, where, assuming Gaussianity, the best model balances fitting the data and minimizing individual uncertainties.')
    iprint('Would you like to run this part? (yes/no)')
    flag=iflag()
    if flag == 'yes':
        iprint('This step saves the output of the Bayesian minimization with the different α values in DIIM_PATH + "/settings/reproduce/alphas". For this step, we used the perturbation factors from the previous step, specifically the ones saved in DIIM_PATH + "/settings/perturbation_factors/perturbation_factors_history_AM_test.npy".')
        cprint("track_alphas(output_path = MODEL_HOME + '/settings/reproduce/alphas',save=True)")
        print('')
        iprint('This can take a bit of time...')
        bayes.track_alphas(output_path = MODEL_HOME + '/settings/reproduce/alphas',save=True)

            
    iprint('Finally, we run the inversion using the best alpha and the optimal perturbation factors to obtain the historical optical constituents along with their uncertainties.')
    iprint('Would you like to run this part? (yes/no)')
    flag=iflag()
    if flag == 'yes':
        iprint('This may take a bit of time...')
        print(""">>>data = rdm.customTensorData(data_path=MODEL_HOME + '/settings/npy_data',which='all',per_day = True,randomice=False)
        >>>perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/settings/perturbation_factors/perturbation_factors_history_AM_test.npy'))[-1].to(torch.float32)

        
        >>>my_device = 'cpu' # the forward computations are not optimal to run with cuda
        >>>constant = rdm.read_constants(file1=MODEL_HOME + '/settings/cte_lambda.csv',file2=MODEL_HOME + '/settings/cte.csv',my_device = my_device)
        
        >>>lr = 0.029853826189179603 #tuned by runing with many lr's
        >>>x_a = torch.zeros(3)
        >>>s_a_ = torch.eye(3)
        >>>s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf
        >>>batch_size = data.len_data
        >>>dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        >>>s_a = s_a_*4.9 #this is the best alpha 
        
        >>>model = fm.Forward_Model(num_days=batch_size).to(my_device)
        >>>model.perturbation_factors = perturbation_factors
        >>>loss = fm.RRS_loss(x_a,s_a,s_e,num_days=batch_size,my_device = my_device)
        >>>optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        
        >>>output = bayes.train_loop(next(iter(dataloader)),model,loss,optimizer,4000,kind='all',\
        num_days=batch_size,constant = constant,perturbation_factors_ = perturbation_factors, scheduler = True)
        
        >>>output_path = MODEL_HOME+'/settings/reproduce/results_AM'
        >>>np.save(output_path + '/X_hat.npy',output['X_hat'])
        >>>np.save(output_path + '/kd_hat.npy',output['kd_hat'])
        >>>np.save(output_path + '/bbp_hat.npy',output['bbp_hat'])
        >>>np.save(output_path + '/RRS_hat.npy',output['RRS_hat'])
        >>>np.save(output_path + '/dates.npy',data.dates) """)
        
        data = rdm.customTensorData(data_path=MODEL_HOME + '/settings/npy_data',which='all',per_day = True,randomice=False)
        perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/settings/perturbation_factors/perturbation_factors_history_AM_test.npy'))[-1].to(torch.float32)
        
        
        my_device = 'cpu'
        constant = rdm.read_constants(file1=MODEL_HOME + '/settings/cte_lambda.csv',file2=MODEL_HOME + '/settings/cte.csv',my_device = my_device)
        
        lr = 0.029853826189179603
        x_a = torch.zeros(3)
        s_a_ = torch.eye(3)
        s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf
        batch_size = data.len_data
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        s_a = s_a_*4.9
        
        model = fm.Forward_Model(num_days=batch_size).to(my_device)
        model.perturbation_factors = perturbation_factors
        loss = fm.RRS_loss(x_a,s_a,s_e,num_days=batch_size,my_device = my_device)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        
        output = bayes.train_loop(next(iter(dataloader)),model,loss,optimizer,4000,kind='all',\
                                  num_days=batch_size,constant = constant,perturbation_factors_ = perturbation_factors, scheduler = True)
        
        output_path = MODEL_HOME+'/settings/reproduce/results_AM'
        np.save(output_path + '/X_hat.npy',output['X_hat'])
        np.save(output_path + '/kd_hat.npy',output['kd_hat'])
        np.save(output_path + '/bbp_hat.npy',output['bbp_hat'])
        np.save(output_path + '/RRS_hat.npy',output['RRS_hat'])
        np.save(output_path + '/dates.npy',data.dates)

    iprint('We can also plot these results, along with the output of the neural network (explained in the forward step), using the module plot_data_lognormal.py')
    cprint("from diimpy import plot_data_lognormal as pdl")
    
    cprint("""pdl.plot_chla(input_data_path = MODEL_HOME + '/experiments/results_bayes_lognormal_VAEparam',\
              figname = MODEL_HOME + '/experiments/chla_lognormal_data_chla_centered.pdf',save=True,date_init = datetime(year=2005,month=1,day=1),\
              statistics=False, num_cols = 1,labels_names=['In situ data','Bayesian MAP output and Uncertainty'],ylim=[],figsize=(17,12),\
              third_data_path = MODEL_HOME + '/settings/VAE_model/results_VAE_VAEparam_chla',log_scale=True)
    """)
    pdl.plot_chla(input_data_path = MODEL_HOME + '/experiments/results_bayes_lognormal_VAEparam',\
              figname = MODEL_HOME + '/experiments/chla_lognormal_data_chla_centered.pdf',save=False,date_init = datetime(year=2005,month=1,day=1),\
              statistics=False, num_cols = 1,labels_names=['In situ data','Bayesian MAP output and Uncertainty'],ylim=[],figsize=(17,12),\
              third_data_path = MODEL_HOME + '/settings/VAE_model/results_VAE_VAEparam_chla',log_scale=True)

    
        
def mcmc_part():

    cprint('from diimpy import sensitivity_analysis_and_mcmc_runs as mcmc')
    print('')
    
    iprint("""First, let's perform a sensitivity analysis of the model's parameters. For this purpose, I computed the Jacobian of the parameters with respect to the different functions kdkd​, bbpbbp​, and RRSRRS​. We will not repeat this step here, as the results are already stored. However, if you wish to recompute it, the code would be:

>>>jacobian_rrs, jacobian_kd, jacobian_bbp = mcmc.compute_jacobians(model, X, mu_z, perturbation_factors, constant=constant)

Here:

    1) model is the forward model,
    2) X is the input data for the model,
    3) mu_z represents the parameters Chl-aChl-a, NAPNAP, and CDOMCDOM,
    4) perturbation_factors specifies the point at which the pointwise Jacobians are computed.

This function calculates the Jacobians at the specified perturbation_factors. """)
            
    iprint('We read the precomputed Jacobians and the pointwise evaluations of the functions. Using these, we generate a box plot to visualize the sensitivity analysis.')
    print('''
    >>>data_dir = MODEL_HOME + '/settings/npy_data'
    
    >>>my_device = 'cpu'
    
    >>>constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cte.csv',my_device = my_device)
    >>>data = rdm.customTensorData(data_path=data_dir,which='all',per_day = True,randomice=False,one_dimensional = False,seed = 1853,device=my_device)
    >>>dataloader = DataLoader(data, batch_size=len(data.x_data), shuffle=False)

    >>>X,Y = next(iter(dataloader))
    
    >>>jacobian_rrs = np.abs(np.load(MODEL_HOME + '/settings/Jacobians/jacobian_rrs_initialparam_lognormalresults.npy')) #drrs/depsiloni
    >>>jacobian_kd = np.abs(np.load(MODEL_HOME + '/settings/Jacobians/jacobian_kd_initialparam_lognormalresults.npy'))
    >>>jacobian_bbp = np.abs(np.load(MODEL_HOME + '/settings/Jacobians/jacobian_bbp_initialparam_lognormalresults.npy'))
    
    >>>chla_hat = np.load( MODEL_HOME + '/experiments/results_bayes_lognormal_unperturbed/X_hat.npy')
    >>>kd_hat = np.load(MODEL_HOME + '/experiments/results_bayes_lognormal_unperturbed/kd_hat.npy')[:,[0,2,4,6,8]]
    >>>bbp_hat = np.load(MODEL_HOME + '/experiments/results_bayes_lognormal_unperturbed/bbp_hat.npy')[:,[0,2,4]]
    >>>rrs_hat = np.load(MODEL_HOME + '/experiments/results_bayes_lognormal_unperturbed/RRS_hat.npy')
    
    >>>perturbation_factors = torch.ones(14)
    >>>sensitivity_boxplot(jacobian_rrs,jacobian_kd,jacobian_bbp,rrs_hat,kd_hat,bbp_hat,perturbation_factors,X,\
                            title='Sensitivity of the parameters with the literature values')
    ''')
    data_dir = MODEL_HOME + '/settings/npy_data'
    
    my_device = 'cpu'
    
    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cte.csv',my_device = my_device)
    data = rdm.customTensorData(data_path=data_dir,which='all',per_day = True,randomice=False,one_dimensional = False,seed = 1853,device=my_device)
    dataloader = DataLoader(data, batch_size=len(data.x_data), shuffle=False)
    
    X,Y = next(iter(dataloader))
    jacobian_rrs = np.abs(np.load(MODEL_HOME + '/settings/Jacobians/jacobian_rrs_initialparam_lognormalresults.npy')) #drrs/depsiloni
    jacobian_kd = np.abs(np.load(MODEL_HOME + '/settings/Jacobians/jacobian_kd_initialparam_lognormalresults.npy'))
    jacobian_bbp = np.abs(np.load(MODEL_HOME + '/settings/Jacobians/jacobian_bbp_initialparam_lognormalresults.npy'))
    
    chla_hat = np.load( MODEL_HOME + '/experiments/results_bayes_lognormal_unperturbed/X_hat.npy')
    kd_hat = np.load(MODEL_HOME + '/experiments/results_bayes_lognormal_unperturbed/kd_hat.npy')[:,[0,2,4,6,8]]
    bbp_hat = np.load(MODEL_HOME + '/experiments/results_bayes_lognormal_unperturbed/bbp_hat.npy')[:,[0,2,4]]
    rrs_hat = np.load(MODEL_HOME + '/experiments/results_bayes_lognormal_unperturbed/RRS_hat.npy')
    
    perturbation_factors = torch.ones(14)
    mcmc.sensitivity_boxplot(jacobian_rrs,jacobian_kd,jacobian_bbp,rrs_hat,kd_hat,bbp_hat,perturbation_factors,X,\
                             title='Sensitivity of the parameters with the literature values')

    iprint('Next, we run an MCMC algorithm with initial conditions close to the output of the alternate minimization (AM). We expect this output to be close to the mode of the distribution, which would result in a small tail to cut from the MCMC chain.')
    iprint('Would you like to run this part? (yes/no)')
    flag=iflag()
    if flag == 'yes':
        
        cprint('>>>mcmc()')
        print('')
        iprint("We will create 20 mcmc runs, and store them in DIIM_PATH + '/experiments/mcmc/runs2/run_' + str(j)+'.npy', with j from 0 to 19. As espected, will take some time...")
        mcmc.mcmc()
    iprint(""" Now we can read our MCMC runs (I did 40 runs), compute autocorrelations to discard self-correlated elements, or plot the final parameters obtained by averaging the final perturbation factors after discarding the tail and the self-correlated elements. Here, we will only plot, but the rest of the code can be found in the sensitivity_analysis_and_mcmc_runs.py script.

Important: We stored the perturbation factors as perturbation_factors_mcmc_mean/std, containing only the uncorrelated values, excluding the tail. However, to plot the next figure, we will plot everything, as we want to show the entire MCMC chain.""")
    
    indexes = rdm.customTensorData(data_path=data_dir,which='all',per_day = True,randomice=True,one_dimensional = False,seed = 1853,device=my_device).train_indexes
    perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/settings/perturbation_factors/perturbation_factors_history_AM_test.npy')[-1]).to(torch.float32)
    perturbation_factors_NN = torch.tensor(np.load(MODEL_HOME + '/settings/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy')[-300:]).to(torch.float32).mean(axis=0)
    perturbation_factors_NN_std = torch.tensor(np.load(MODEL_HOME + '/settings/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy')[-300:]).to(torch.float32).std(axis=0)
    chla_hat = torch.tensor(np.load( MODEL_HOME + '/experiments/results_bayes_lognormal_logparam/X_hat.npy')[:,[0,2,4]]).to(torch.float32).unsqueeze(1)[indexes]
    constant_values = np.array([constant['dCDOM'],constant['sCDOM'],5.33,0.45,constant['Theta_min'],constant['Theta_o'],constant['beta'],constant['sigma'],0.005])

    num_runs = 40
    mcmc_runs = np.empty((num_runs,3000,14))


    for i in range(num_runs):
        mcmc_runs[i] = np.load(MODEL_HOME + '/experiments/mcmc/run_' + str(i)+'.npy')

    #mcmc_runs = mcmc_runs[:,2000:,:]
    mcmc_runs_mean, mcmc_runs_std = np.empty((14)),np.empty((14))

    mcmc_runs = mcmc_runs[:,:,5:] * constant_values

    mcmc_runs_mean = np.mean(mcmc_runs,axis=0)
    mcmc_percentile_2_5 = np.percentile(mcmc_runs,2.5,axis=0)
    
    mcmc_percentile_98_5 = np.percentile(mcmc_runs,98.5,axis=0)
    mcmc_percentile_16 = np.percentile(mcmc_runs,16,axis=0)
    mcmc_percentile_84 = np.percentile(mcmc_runs,84,axis=0)
    
    
    fig_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    names = ['$d_{\mathrm{CDOM}}$ [$\mathrm{m}^2(\mathrm{mgCDOM})^{-1}$]','$S_{\mathrm{CDOM}}$ [nm]','$Q_a$','$Q_b$',\
             '$\Theta^{\mathrm{min}}_{\mathrm{chla}}$ [$\mathrm{mgChla}\mathrm{(mgC)}^{-1}$]','$\Theta^{\mathrm{0}}_{\mathrm{chla}}$  [$\mathrm{mgChla}\mathrm{(mgC)}^{-1}$]',\
             '$\\beta$ [$\mathrm{mmol}\mathrm{m}^{-2}\mathrm{s}^{-1}$]','$\sigma$  [$\mathrm{mmol}\mathrm{m}^{-2}\mathrm{s}^{-1}$]','$b_{b,\mathrm{NAP}}$']
    which = 3
    fig,axs = plt.subplots(ncols=2,nrows=1,layout='constrained',width_ratios=[3/4,1/4])

    axs[0].axhline( y = constant_values[which],linestyle='--',label='original value',color='black')
    axs[0].axhline( y = mcmc_runs_mean[500:,which].mean(),linestyle='dashdot',label='mcmc relaxation mean',color='blue')
    axs[0].axhline( y = constant_values[which]*perturbation_factors_NN[5+which],linestyle='dashdot',label='CVAE result',color='#f781bf')
    
    

    ynew = scipy.ndimage.uniform_filter1d(mcmc_runs_mean[:,which], size=100)
    axs[0].plot(ynew,label='mcmc')

    ynew1 = scipy.ndimage.uniform_filter1d(mcmc_percentile_2_5[:,which], size=100)
    ynew2 = scipy.ndimage.uniform_filter1d(mcmc_percentile_98_5[:,which], size=100)
    axs[0].fill_between(range(3000), ynew1, ynew2,color='gray',zorder = 0.1,alpha=0.6,label = '95% confidence interval of mcmc')
    
    ynew1 = scipy.ndimage.uniform_filter1d(mcmc_percentile_16[:,which], size=100)
    ynew2 = scipy.ndimage.uniform_filter1d(mcmc_percentile_84[:,which], size=100)
    axs[0].fill_between(range(3000), ynew1, ynew2,color='#377eb8',zorder = 0.1,alpha=0.8,label = '63% confidence interval of mcmc')
    


    axs[0].set_xlim(0,3000)
    axs[0].text(-0.1,1.05,'(a)',transform = axs[0].transAxes,fontsize=20)
    axs[0].set_ylabel(names[which])
    axs[0].set_xlabel('Iterations')

    axs[0].legend()

    data = mcmc_runs[:,2000:,which] #cuting the tail
    data = data[:,::280].flatten() # uncorrelated values
    data = data
    
    
    n,bins,patches = axs[1].hist(data,orientation='horizontal',bins=40,edgecolor='black',color='#377eb8',alpha=0.6,density=True)
    (mu, sigma) = scipy.stats.norm.fit(data)
    axs[1].plot(scipy.stats.norm.pdf(bins,loc=mu,scale=sigma),bins , linewidth=2,color='black',label='normal distribution\nmean: {:.3f}\nstd: {:.3f}'.format(mu,sigma))
    axs[1].legend()
    axs[1].text(-0.1,1.05,'(b)',transform = axs[1].transAxes,fontsize=20)
    axs[1].set_xlabel('Probability density')

    axs[1].set_yticks([])
    axs[0].set_ylim(*axs[1].get_ylim())
    plt.show()
    plt.close()
    mcmc_runs = mcmc_runs[:,2000:,:]

    iprint('We can also plot the lambda dependent parameters,')
    cprint("mcmc.plot_constants_2(perturbation_path = MODEL_HOME + '/settings/perturbation_factors',vae_name = 'perturbation_factors_history_CVAE_chla_centered.npy')")
    print('')
    mcmc.plot_constants_2(perturbation_path = MODEL_HOME + '/settings/perturbation_factors',vae_name = 'perturbation_factors_history_CVAE_chla_centered.npy')
    iprint('Or even create a table with the statistics of the parameters (this table is in latex table format, use plot=False,table=True to create it as in the paper),')
    cprint("constant_values = np.array([constant['dCDOM'],constant['sCDOM'],5.33,0.45,constant['Theta_min'],constant['Theta_o'],constant['beta'],constant['sigma'],0.005])")
    cprint('mcmc.parameters_statistics(num_runs,mcmc_runs,names,correlation_lenght=280,plot=True,table=False)')
    constant_values = np.array([constant['dCDOM'],constant['sCDOM'],5.33,0.45,constant['Theta_min'],constant['Theta_o'],constant['beta'],constant['sigma'],0.005])
    mcmc.parameters_statistics(num_runs,mcmc_runs,names,correlation_lenght=280,plot=True,table=False,constant_values=constant_values,perturbation_factors = perturbation_factors,perturbation_factors_NN = perturbation_factors_NN,fig_labels=fig_labels)

def NN_part():
    iprint(''' A different approach is to use the SGVB framework (see the paper, section 4.5). It consists of training a probabilistic neural network with a latent variable structure. The final architecture and parameters for the neural network were tuned using Ray Tune. As described in the README file, this framework consists of three parts.

The first part is a neural network that maps from the input data to the historical data (intended as a dimensionality reduction step). The architecture is as follows:

    Input layer: 17 units (same input as the Remote Sensing Reflectance, excluding the wavelengths),
    Two hidden layers with 20 and 22 units,
    Output layer: 9 units.

This is a fully connected neural network, with the non-linearity defined by the CELU function (>>>torch.CELU(alpha=0.8978238833058)).

The state dictionary of the neural network after training is saved in DIIM_PATH + "/settings/VAE_model/model_first_part.pt".''')
    
    iprint('We used a function to initialize the neural network with an arbitrary number of input, output, and hidden layers. The script to initialize the first part is as follows:')
    cprint(""" model_first_layer = cvae_one.NN_first_layer(precision = torch.float32,input_layer_size=17,output_layer_size=9,\
    number_hiden_layers = 1,dim_hiden_layers = 20,dim_last_hiden_layer = 22,alpha=0.8978238833058)
    model_first_layer.load_state_dict(torch.load(HOME_PATH + '/settings/VAE_model/model_first_part.pt'))
 """)
    model_first_layer = cvae_one.NN_first_layer(precision = torch.float32,input_layer_size=17,output_layer_size=9,\
    number_hiden_layers = 1,dim_hiden_layers = 20,dim_last_hiden_layer = 22,alpha=0.8978238833058)
    model_first_layer.load_state_dict(torch.load(HOME_PATH + '/settings/VAE_model/model_first_part.pt'))

    print('')
    iprint('The output of the first neural network is the input of a second neural network, composed of a mean_ function and a std_ function (see Figure 2 from the paper), both fully connected with CELU nonlinearities. The mean part has four hidden layers of dimension 18, and one with dimension 19, an output layer of size 3, and CELU activation functions with parameter alpha = 1.3822406736258. The std part has 2 hidden layers with size 13 and one with size 11, with an output layer of size 9, and CELU activation functions with parameter alpha = 0.7414694152899. The state dictionary of the trained neural network is in DIIM_PATH + "/settings/VAE_model/model_second_part_chla_centered.pt", and the parameters for the configuration are in MODEL_HOME + "/settings/VAE_model/model_second_part_final_config.pt"')
    iprint('To use this function, first read the configuration file, then initialize the neural network, and finally load the state dictionary. An example of how to use it is:')

    print("""

    >>>data_dir = MODEL_HOME + '/settings'
        
    >>>best_result_config = torch.load(MODEL_HOME + '/settings/VAE_model/model_second_part_final_config.pt')

    >>>batch_size = int(best_result_config['batch_size'])
    >>>number_hiden_layers_mean = best_result_config['number_hiden_layers_mean']
    >>>dim_hiden_layers_mean = best_result_config['dim_hiden_layers_mean']
    >>>dim_last_hiden_layer_mean = best_result_config['dim_last_hiden_layer_mean']
    >>>alpha_mean = best_result_config['alpha_mean']
    >>>number_hiden_layers_cov = best_result_config['number_hiden_layers_cov']
    >>>dim_hiden_layers_cov = best_result_config['dim_hiden_layers_cov']
    >>>dim_last_hiden_layer_cov = best_result_config['dim_last_hiden_layer_cov']
    >>>alpha_cov = best_result_config['alpha_cov']
    >>>lr = best_result_config['lr']
    >>>betas1 = best_result_config['betas1'] 
    >>>betas2 = best_result_config['betas2']
    >>>dk_alpha = best_result_config['dk_alpha']

    >>>my_device = 'cpu'

    >>>constant = rdm.read_constants(file1=MODEL_HOME + '/settings/cte_lambda.csv',file2=MODEL_HOME+'/settings/cte.csv',my_device = my_device)
    >>>data = rdm.customTensorData(data_path=data_dir+'/npy_data',which='all',per_day = False,randomice=False,one_dimensional = True,seed = 1853,device=my_device,normilized_NN='scaling')

    >>>dataloader = DataLoader(data, batch_size=len(data.x_data), shuffle=False)

    >>>model = cvae_two.NN_second_layer(output_layer_size_mean=3,number_hiden_layers_mean = number_hiden_layers_mean,\
                           dim_hiden_layers_mean = dim_hiden_layers_mean,alpha_mean=alpha_mean,dim_last_hiden_layer_mean = dim_last_hiden_layer_mean,\
                           number_hiden_layers_cov = number_hiden_layers_cov,\
                           dim_hiden_layers_cov = dim_hiden_layers_cov,alpha_cov=alpha_cov,dim_last_hiden_layer_cov = dim_last_hiden_layer_cov,x_mul=data.x_mul,x_add=data.x_add,\
                           y_mul=data.y_mul,y_add=data.y_add,constant = constant,model_dir = HOME_PATH + '/settings/VAE_model').to(my_device)

    >>>model.load_state_dict(torch.load(MODEL_HOME + '/settings/VAE_model/model_second_part_chla_centered.pt'))
    >>>X,Y = next(iter(dataloader))
    
    >>>z_hat,cov_z,mu_z,kd_hat,bbp_hat,rrs_hat = model(X)

    >>>plt.plot(z_hat[:,0].clone().detach())
    >>>plt.show()
    """)
    
    data_dir = MODEL_HOME + '/settings'
        
    best_result_config = torch.load(MODEL_HOME + '/settings/VAE_model/model_second_part_final_config.pt')

    batch_size = int(best_result_config['batch_size'])
    number_hiden_layers_mean = best_result_config['number_hiden_layers_mean']
    dim_hiden_layers_mean = best_result_config['dim_hiden_layers_mean']
    dim_last_hiden_layer_mean = best_result_config['dim_last_hiden_layer_mean']
    alpha_mean = best_result_config['alpha_mean']
    number_hiden_layers_cov = best_result_config['number_hiden_layers_cov']
    dim_hiden_layers_cov = best_result_config['dim_hiden_layers_cov']
    dim_last_hiden_layer_cov = best_result_config['dim_last_hiden_layer_cov']
    alpha_cov = best_result_config['alpha_cov']
    lr = best_result_config['lr']
    betas1 = best_result_config['betas1'] 
    betas2 = best_result_config['betas2']
    dk_alpha = best_result_config['dk_alpha']

    my_device = 'cpu'

    constant = rdm.read_constants(file1=MODEL_HOME + '/settings/cte_lambda.csv',file2=MODEL_HOME+'/settings/cte.csv',my_device = my_device)
    data = rdm.customTensorData(data_path=data_dir+'/npy_data',which='all',per_day = False,randomice=False,one_dimensional = True,seed = 1853,device=my_device,normilized_NN='scaling')

    dataloader = DataLoader(data, batch_size=len(data.x_data), shuffle=False)

    model = cvae_two.NN_second_layer(output_layer_size_mean=3,number_hiden_layers_mean = number_hiden_layers_mean,\
                           dim_hiden_layers_mean = dim_hiden_layers_mean,alpha_mean=alpha_mean,dim_last_hiden_layer_mean = dim_last_hiden_layer_mean,\
                           number_hiden_layers_cov = number_hiden_layers_cov,\
                           dim_hiden_layers_cov = dim_hiden_layers_cov,alpha_cov=alpha_cov,dim_last_hiden_layer_cov = dim_last_hiden_layer_cov,x_mul=data.x_mul,x_add=data.x_add,\
                           y_mul=data.y_mul,y_add=data.y_add,constant = constant,model_dir = HOME_PATH + '/settings/VAE_model').to(my_device)

    model.load_state_dict(torch.load(MODEL_HOME + '/settings/VAE_model/model_second_part_chla_centered.pt'))
    X,Y = next(iter(dataloader))
    
    z_hat,cov_z,mu_z,kd_hat,bbp_hat,rrs_hat = model(X)

    plt.plot(z_hat[:,0].clone().detach())
    plt.show()

    iprint('To tune the parameters, the function used was >>>cvae_two.explore_hyperparameters(). Once Ray Tune explores the possible parameters, the function >>>save_cvae_first_part() loads them and trains the neural network with all the training data (to tune the parameters, the neural network was trained with 90% of the training data). The trained state dictionary is then stored in DIIM_PATH + "/settings/VAE_model/model_second_part_chla_centered.pt. The names are the same for training the first part and the second part of the neural network. Finally, the output can be stored.')
    iprint('Finally, you can save the results to use them in the future.')
    iprint('Would you like to run this part? (yes/no)')
    flag=iflag()
    if flag == 'yes':


        print("""
        mu_z = mu_z* data.y_mul[0] + data.y_add[0]
        cov_z = torch.diag(data.y_mul[0].expand(3)).T @ cov_z @ torch.diag(data.y_mul[0].expand(3)) 
        kd_hat = kd_hat * data.y_mul[1:6] + data.y_add[1:6]
        bbp_hat = bbp_hat * data.y_mul[6:] + data.y_add[6:]
        rrs_hat = rrs_hat * data.x_mul[:5] + data.x_add[:5]
        X = model.rearange_RRS(X)
        
        cvae_two.save_var_uncertainties(model.Forward_Model,X,mu_z,cov_z,rrs_hat,constant=constant,dates = data.dates,save_path =MODEL_HOME + '/settings/reproduce/results_VAE_VAEparam_chla')
        """)
        print('')
        iprint('The computation of the covariance matrix may take a bit of time ...')
        mu_z = mu_z* data.y_mul[0] + data.y_add[0]
        cov_z = torch.diag(data.y_mul[0].expand(3)).T @ cov_z @ torch.diag(data.y_mul[0].expand(3)) 
        kd_hat = kd_hat * data.y_mul[1:6] + data.y_add[1:6]
        bbp_hat = bbp_hat * data.y_mul[6:] + data.y_add[6:]
        rrs_hat = rrs_hat * data.x_mul[:5] + data.x_add[:5]
        X = model.rearange_RRS(X)
        
        cvae_two.save_var_uncertainties(model.Forward_Model,X,mu_z,cov_z,rrs_hat,constant=constant,dates = data.dates,save_path =MODEL_HOME + '/settings/reproduce/results_VAE_VAEparam_chla')


    
    
if __name__ ==  '__main__':
    try:
        args = argument()

        file_ = open(HOME_PATH + '/settings/reproduce/interactive_cat')
        interactive_cat = file_.read()
        print(interactive_cat)
        iprint('Welcome to the interactive script to reproduce the results from DOI: https://doi.org/10.5194/gmd-2024-174. At any time type ctl+c to stop the interactive experience')
        
        iprint('The paper "Data-Informed Inversion Model (DIIM): Framework to Retrieve Marine Optical Constituents in the BOUSSOLE Site Using a Three-Stream Irradiance Model" uses a one-dimensional bio-optical model and historical data to test two different approaches for retrieving marine optical constituents from Remote Sensing Reflectance data. The goal is to retrieve values along with their uncertainties, as well as to optimize the model using data from past measurements.',delay = 0.002)
    
        if args.a:
            bayesian_part()    
            print(interactive_cat)
            iprint('Nice, we have our first result, but we will not use this one. We would like to have a measure of the uncertainty of the perturbation factors also. To do so, we will use a montecarlo algorithm. Script in file sensitivity_analysis_and_mcmc_runs.py')
            mcmc_part()
            print(interactive_cat)
            NN_part()
            
        if args.b:
            bayesian_part()
        if args.m:
            iprint('Assuming we used Alternate minimization to make create a first approximation of the optimal permutation factors, and of the optical constituents, we would like to have a measure of the uncertainty of the perturbation factors also. To do so, we will use a montecarlo algorithm. Script in file sensitivity_analysis_and_mcmc_runs.py')
            mcmc_part()
        if args.n:
            NN_part()
        print(interactive_cat)
        iprint('Thank you for reading these interactive gide on how to reproduce the results from the paper ^^')
                
            
    except KeyboardInterrupt:
        print('')
        print('Are you sure you want to stop this experience :( (yes/no)')
        flag=iflag()
        if flag == 'yes':
            print(':(')
            sys.exit()
        else:
            print('^.^')
            sys.exit()
        
        
