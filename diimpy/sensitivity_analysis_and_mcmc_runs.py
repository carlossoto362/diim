
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import scipy
from scipy import stats
from torch.utils.data import DataLoader,random_split
import time
import seaborn as sb
from diimpy.Forward_module import evaluate_model_class,Forward_Model,RRS_loss,OBS_loss
from diimpy.read_data_module import customTensorData ,read_constants
from torch.utils.data import DataLoader
from multiprocess.pool import Pool
import matplotlib.colors as mcolors
from diimpy.CVAE_model_part_two import NN_second_layer
import os
import sys
from mpi4py import MPI

if 'DIIM_PATH' in os.environ:
    MODEL_HOME = HOME_PATH =  os.environ["DIIM_PATH"]
else:
    
    print("Missing local variable DIIM_PATH. \nPlease add it with '$:export DIIM_PATH=path/to/diim'.")
    sys.exit()

def compute_jacobians(Forward_Model_, X, chla_hat_mean,perturbation_factors, constant = None):
    parameters_eval = perturbation_factors
    evaluate_model = evaluate_model_class(Forward_Model_,X,constant=constant,which_parameters='perturbations',chla=chla_hat_mean)
        
    jacobian_rrs = torch.autograd.functional.jacobian(evaluate_model.model_der,inputs=(parameters_eval))
    jacobian_kd = torch.autograd.functional.jacobian(evaluate_model.kd_der,inputs=(parameters_eval))
    jacobian_bbp = torch.autograd.functional.jacobian(evaluate_model.bbp_der,inputs=(parameters_eval))

    

    return jacobian_rrs,jacobian_kd,jacobian_bbp

def sensitivity_boxplot(jacobian_rrs,jacobian_kd,jacobian_bbp,rrs_hat,kd_hat,bbp_hat,perturbation_factors,X,\
                            title='Sensitivity of the parameters near the AM solution',lims=[-1e-1,13]):

        rrs_normal = rrs_hat.reshape((rrs_hat.shape[0],5,1))
        epsilons = (np.tile(np.repeat(1/perturbation_factors,5).reshape(14,5).T,(X.shape[0],1)).reshape((X.shape[0],5,14)))
        normalization = rrs_normal * epsilons
        jacobian_rrs = np.mean(jacobian_rrs/normalization,axis=1)

        kd_normal = kd_hat.reshape((kd_hat.shape[0],5,1))
        normalization = kd_normal * epsilons
        jacobian_kd = np.mean(jacobian_kd/normalization,axis=1)


        bbp_normal = bbp_hat.reshape((bbp_hat.shape[0],3,1))
        epsilons = (np.tile(np.repeat(1/perturbation_factors,3).reshape(14,3).T,(X.shape[0],1)).reshape((X.shape[0],3,14)))
        normalization = bbp_normal * epsilons
        jacobian_bbp = np.mean(jacobian_bbp/normalization,axis=1)


        xticks = ['${a_{phy}}$','$\delta_{b_{phy,T}}$','${b_{phy,Int}}$','${b_{b,phy,T}}$','${b_{b,phy,Int}}$','${d_{\mathrm{CDOM}}}$','${S_{\mathrm{CDOM}}}$','${Q_a}$','${Q_b}$',\
                  '${\Theta^{\mathrm{min}}_{\mathrm{chla}}}$','${\Theta^{\mathrm{0}}_{\mathrm{chla}}}$',\
                  '${\\beta}$','${\sigma}$','${b_{b,\mathrm{NAP}}}$']


    
        jacobian_rrs_dataframe = pd.DataFrame()
        for i in range(14):
            jacobian_rrs_dataframe[xticks[i]] = jacobian_rrs[:,i]

        jacobian_kd_dataframe = pd.DataFrame()
        for i in range(14):
            jacobian_kd_dataframe[xticks[i]] = jacobian_kd[:,i]

        jacobian_bbp_dataframe = pd.DataFrame()
        for i in range(14):
            jacobian_bbp_dataframe[xticks[i]] = jacobian_bbp[:,i]


        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(15*0.7, 9*0.65),
                                layout="constrained")
        sb.boxplot(data=jacobian_rrs_dataframe,ax=axs[0],whis=(10, 90),color='#89b7dc',fliersize=1,widths=0.3,saturation=1,linewidth=1.5)
        sb.boxplot(data=jacobian_kd_dataframe,ax=axs[1],whis=(10, 90),color='#89b7dc',fliersize=1,widths=0.3,saturation=1)
        sb.boxplot(data=jacobian_bbp_dataframe,ax=axs[2],whis=(10, 90),color='#89b7dc',fliersize=1,widths=0.3,saturation=1)
    
    

        axs[0].axes.xaxis.set_ticklabels([])
        axs[0].set_ylabel('$ \partial( \log{ R_{RS} }) \partial (\log{\delta_i})^{-1} $',fontsize=16)
        axs[0].tick_params(axis='y', labelsize=14)
        axs[0].text(1-0.04,0.9,'(a)',transform = axs[0].transAxes,fontsize=15)
        axs[0].set_yscale('symlog')
        yticks = np.linspace(lims[0],lims[1]+1,12)
        axs[0].set_yticks(yticks,labels=['{:.1e}'.format(yticks[i]) if i in [0,1,2,8] else '' for i in range(len(yticks))])
        axs[0].yaxis.grid(True,alpha=0.4)
        axs[0].set_ylim(*lims)
        
        axs[1].axes.xaxis.set_ticklabels([])
        axs[1].set_ylabel('$ \partial (\log{ kd }) \partial (\log{\delta_i})^{-1} $',fontsize=16)
        axs[1].tick_params(axis='y', labelsize=14)
        axs[1].text(1-0.04,0.9,'(b)',transform = axs[1].transAxes,fontsize=15)
        axs[1].set_yscale('symlog')
        axs[1].set_yticks(yticks,labels=['{:.1e}'.format(yticks[i]) if i in [0,1,2,8] else '' for i in range(len(yticks))])
        axs[1].yaxis.grid(True,alpha=0.4)
        axs[1].set_ylim(*lims)
    
        axs[2].tick_params(axis='x', labelsize=14)
        axs[2].set_ylabel('$ \partial (\log{ b_{b,p} }) \partial (\log{\delta_i})^{-1} $',fontsize=16)
        axs[2].tick_params(axis='y', labelsize=14)
        axs[2].text(1-0.04,0.9,'(c)',transform = axs[2].transAxes,fontsize=15)
        axs[2].set_yscale('symlog')
        axs[2].set_yticks(yticks,labels=['{:.1e}'.format(yticks[i]) if i in [0,1,2,8] else '' for i in range(len(yticks))])
        axs[2].yaxis.grid(True,alpha=0.4)
        axs[2].set_ylim(*lims)
        axs[2].text(-0.05,-0.15,'$\delta_i,i= $',transform = axs[2].transAxes,fontsize=15)
                
        
        plt.show()


class evaluate_model():

    def __init__(self,data_path = MODEL_HOME + '/settings/npy_data',iterations=10, my_device = 'cpu',precision=torch.float32,constant=None):

        self.precision = precision
        self.data = customTensorData(data_path=data_path,which='train',per_day = False,randomice=False,seed=1853,precision=self.precision)
        self.dates = self.data.dates
        self.my_device = 'cpu'

        self.constant=constant
    
        self.x_a = torch.zeros(3)
        self.s_a = torch.eye(3)*1.13
        self.s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf

        self.lr = 0.029853826189179603
        self.batch_size = self.data.len_data
        self.dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=False)
        self.model = Forward_Model(num_days=self.batch_size,precision=self.precision).to(my_device)

        self.loss = RRS_loss(self.x_a,self.s_a,self.s_e,num_days=self.batch_size,my_device = self.my_device,precision=self.precision)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.iterations = iterations
        self.X,self.Y = next(iter(self.dataloader))

    def model_parameters_update(self,perturbation_factors):        
        self.model.perturbation_factors = perturbation_factors

    def model_chla_init(self,chla_hat):
        state_dict = self.model.state_dict()
        
        state_dict['chparam'] = chla_hat
        self.model.load_state_dict(state_dict)

    def step(self,iteration):
        RRS = self.model(self.X[:,:,1:],constant = self.constant)
        loss = self.loss(self.X[:,:,0],RRS,self.model.state_dict()['chparam'])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss_iter = loss
        #self.scheduler.step(loss)
        
    def predict(self):
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        list(map(self.step,range(self.iterations)))

        
    def return_chla(self):
        return self.model.state_dict()['chparam'].clone().detach()
    
class mcmc_class():
    def __init__(self,chla_hat,perturbation_factors = torch.ones(14),proposal_variance = 1,constant = None,num_iterations = 100,precision=torch.float32):
        
        self.precision = precision
        self.current_position = perturbation_factors + torch.randn(14)*0.05
        self.proposal_variance = proposal_variance
        self.model = evaluate_model(precision=self.precision,constant=constant)
        self.X, self.Y_nan  = next(iter(self.model.dataloader))
        
        self.Y = torch.masked_fill(self.Y_nan,torch.isnan(self.Y_nan),0)
        self.model.model_parameters_update(self.current_position)
        self.model.model_chla_init(chla_hat)
        self.model.predict()
        self.model.iterations = 200
        self.evaluate_model = evaluate_model_class(model = None,X = self.X[:,:,1:],constant = constant)
        self.loss = OBS_loss(precision=precision,normalization_values = torch.tensor(self.model.data.y_std))

        self.history = torch.empty((num_iterations,14))

    def forward_sampling(self):
        return self.proposal_variance * torch.randn(14) + self.current_position

    def likelihood(self,position,chla):
        kd = self.evaluate_model.kd_der(parameters_eval = chla, perturbation_factors_ = self.current_position)
        bbp = self.evaluate_model.bbp_der(parameters_eval = chla, perturbation_factors_ = self.current_position)
            
        pred = torch.empty((chla.shape[0],9))
        pred[:,0] = chla[:,0,0]
        pred[:,1:6] = torch.masked_fill(kd,torch.isnan(self.Y_nan[:,1:6]),0)
        pred[:,6:] = torch.masked_fill(bbp,torch.isnan(self.Y_nan[:,6:]),0)

        loss = self.loss(self.Y,pred,self.Y_nan)
        return -(0.5)*loss*pred.shape[0]

    def likelihood_profile(self,initial_perturbation_factors=None,perturbation_profiled=-1,values_profiled=np.linspace(0.1,1.9,200),iterations=100):
        self.model.iterations = iterations
        likelihood_profile = np.empty((values_profiled.shape[0],3))
        if type(initial_perturbation_factors) != type(None):
            self.current_position = initial_perturbation_factors
            self.model.model_parameters_update(self.current_position)
        
        for k,profile in enumerate(values_profiled):
            init_time = time.time()
            self.current_position[perturbation_profiled] = profile
            self.model.model_parameters_update(self.current_position)

            self.model.predict()
            
            chla_current = self.model.return_chla()
            likelihood_profile[k,0] = (self.likelihood(self.current_position,chla_current).clone().detach().numpy())
            likelihood_profile[k,1] = likelihood_profile[k,0] -(0.5)*self.model.loss_iter
            likelihood_profile[k,2] = likelihood_profile[k,1] - ((0.25)*(self.current_position - np.ones(14))**2).mean()
            print(profile,likelihood_profile[k],init_time - time.time())
            
        return likelihood_profile
                
    def step(self,iteration):
            
        chla_current = self.model.return_chla()
        log_likelihood_current = self.likelihood(self.current_position,chla_current)

        new_state = self.forward_sampling()
        self.model.model_parameters_update(new_state)
        self.model.predict()
        chla_new = self.model.return_chla()
        log_likelihood_new = self.likelihood(new_state,chla_new)

        rho = (log_likelihood_new-log_likelihood_current)
        rho = torch.exp(rho)
        if rho >= 1:
            self.current_position = new_state
        elif torch.rand(1) < rho:
            self.current_position = new_state
        else:
            self.model.model_parameters_update(self.current_position)
            self.model.model_chla_init(chla_current)
        self.history[iteration] = self.current_position

def saving_mcmc(input_):

    j = input_[0]
    perturbation_factors = input_[1]
    constant = input_[2]
    chla_hat = input_[3]
    output_path = input_[4]
    print('saving run', j,'...')
        
    init_time = time.time()
    num_iterations = 3000
    mcmc = mcmc_class(chla_hat,perturbation_factors = perturbation_factors,proposal_variance = 0.002,constant = constant,num_iterations = num_iterations,precision=torch.float64)
            
    resulting_mcmc = list(map(mcmc.step,np.arange(num_iterations)))
        
        
    np.save(output_path + '/run_' + str(j)+'.npy',mcmc.history.numpy())
    print(output_path + '/run_' + str(j)+'.npy saved, time', (time.time() - init_time) )

def profiling_likelihood(perturbation_factors_path,constant_path1,constant_path2,chla_hat_path,output_path):
    my_device = 'cpu'
    precision = torch.float64
    perturbation_factors = torch.tensor(np.load(perturbation_factors_path)).to(precision)
    constant = read_constants(file1=constant_path1,file2=constant_path2,my_device = my_device,precision=precision)
    chla_hat = np.load(chla_hat_path+'/X_hat.npy')
    mu_z = torch.tensor(chla_hat[:,::2],dtype=precision)
    train_labels = np.load(MODEL_HOME+'/settings/npy_data/train_labels.npy')
    train_labels= np.sort(train_labels)
    chla_hat = mu_z[train_labels].unsqueeze(1)
    mcmc = mcmc_class(chla_hat,perturbation_factors = perturbation_factors,proposal_variance = 0.002,constant = constant,num_iterations = 1000,precision=torch.float64)
    likelihood_profile = mcmc.likelihood_profile(initial_perturbation_factors=perturbation_factors,perturbation_profiled=-1,values_profiled=np.linspace(0.1,1.9,100),iterations=500)
    np.save(output_path,likelihood_profile)
    
    
    
def correlation_matrix(num_runs,mcmc_runs,plot=True,table=False):
    correlation_lenght_use = 0
    for which in range(14):

        correlation = np.empty((num_runs,500))
        for i in range(num_runs):
            correlation[i] = autocorr(mcmc_runs[i,:,which],range(500))
        correlation = np.mean(correlation,axis=0)

        for correlation_lenght in range(500):
            if correlation[correlation_lenght]<0.2:
                break
        if correlation_lenght > correlation_lenght_use:
            correlation_lenght_use = correlation_lenght
    print(correlation_lenght)
    data = mcmc_runs[:,::correlation_lenght,:]

    mcmc_runs_ = np.empty((280,14))
    
    for i in range(14):
        mcmc_runs_[:,i] = data[:,:,i].flatten()

    mcmc_runs_dataframe = pd.DataFrame()
    ticks = ['${a_{PH}}$','${b_{phy,T}}$','${b_{phy,\mathrm{Int}}}$','${b_{b,phy,T}}$','${b_{b,phy,\mathrm{Int}}}$','${d_{\mathrm{CDOM}}}$','${S_{\mathrm{CDOM}}}$','${Q_a}$','${Q_b}$',\
                  '${\Theta^{\\text{min}}_{\mathrm{chla}}}$','${\Theta^{\mathrm{0}}_{\mathrm{chla}}}$',\
                  '${\\beta}$','${\sigma}$','${b_{b,\mathrm{NAP}}}$']
    for i,tick in enumerate(ticks):
        mcmc_runs_dataframe[tick] = mcmc_runs_[:,i]
    
    correlation_matrix = mcmc_runs_dataframe.corr()
    print(correlation_matrix)

    if plot == True:
    
        fig,ax = plt.subplots(layout='constrained')
        colors1 = plt.cm.Greys_r(np.linspace(0.,1, 128))
        colors2 = plt.cm.Blues(np.linspace(0., 1, 128))
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
        sb.heatmap(correlation_matrix, cmap=mymap, annot=True,vmin=-1,ax=ax)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        plt.show()
        plt.close()
    if table == True:

        print("$\\delta_i,i=$ & "+" & ".join(ticks) + "\\\\\\hline")
        for i in range(14):
            print(ticks[i] + " & " + " & ".join(["{:.2f}".format(correlation_matrix[ticks[i]][ticks[j]]) for j in range(i+1)]) + "".join(["&"]*(13-i)) +  "\\\\")
        
    

    
def autocorr(x,lags):
        '''numpy.corrcoef, partial'''

        corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
        corr = [1. if np.isnan(c) else c for c in corr ] #worst case scenario, they are correlated. 
        return np.array(corr)

def parameters_statistics(num_runs,mcmc_runs,names,correlation_lenght=280,plot=True,table=False,constant_values=None,perturbation_factors = None,perturbation_factors_NN = None,fig_labels = None,perturbation_factors_NN_std = None,plots_path = MODEL_HOME + '/settings/reproduce/plots'):

    if plot==True:
        fig,axs = plt.subplots(ncols=3,nrows=3,layout='constrained',figsize=(30,20))
        for which,ax in enumerate(axs.flatten()):

            data = mcmc_runs[:,::correlation_lenght,which].flatten()
    
            (mu, sigma) = scipy.stats.norm.fit(data)
            n, bins, patches = ax.hist(data,bins=40,density=True,edgecolor='black',color='#377eb8',alpha=0.6)

            ks = scipy.stats.kstest((data - data.mean())/data.std(),scipy.stats.norm.cdf)


            
            ax.plot(bins, scipy.stats.norm.pdf(bins,loc=mu,scale=sigma), linewidth=2,color='black',label='mean value: {:.4f}\nstd value: {:.4f}\nKS statistic: {:.4f}\np-value: {:.4f}'.format(mu,sigma,ks.statistic,ks.pvalue))

        
            ax.axvline(x=constant_values[which],label= 'original values: {:.4f}'.format(constant_values[which]),color='gray',linestyle='--')
            ax.axvline( x = constant_values[which]*perturbation_factors[5+which],linestyle='dashdot',label='AM result: {:.4f}'.format(constant_values[which]*perturbation_factors[5+which]),color='blue')
            ax.axvline( x = constant_values[which]*perturbation_factors_NN[5+which],linestyle='dashdot',label='CVAE result: {:.4f}'.format(constant_values[which]*perturbation_factors_NN[5+which]),color='#f781bf')
            ax.set_xlabel(names[which],fontsize='15')
            ax.text(-0.1,1.05,fig_labels[which],transform = ax.transAxes,fontsize='20')

            y_lims = ax.get_ylim()
            x_lims = ax.get_xlim()
            ax.fill_between(np.linspace(mu-sigma,mu+sigma,20), [y_lims[0]]*20, [y_lims[1]]*20,color='gray',zorder = 0.1,alpha=0.8,label='63% confidence interval')
            ax.set_ylim(y_lims)
            ax.tick_params(axis='y', labelsize=15)
            ax.tick_params(axis='x', labelsize=15)
            ax.set_xlim((x_lims[0] - (x_lims[1] - x_lims[0] )*0.3,x_lims[1]))
            ax.legend(fontsize="10")
        plt.savefig(plots_path + '/mcmc_parameter_statistics.pdf')
    if table == True:
        mu = np.empty(9)
        sigma = np.empty(9)
        ks_value = np.empty(9)
        ks_pvalue = np.empty(9)

        for which in range(9):
            data = mcmc_runs[:,::correlation_lenght,which].flatten()
            (mu[which], sigma[which]) = scipy.stats.norm.fit(data)
            ks = scipy.stats.kstest((data - data.mean())/data.std(),scipy.stats.norm.cdf)
            ks_value[which] = ks.statistic
            ks_pvalue[which] = ks.pvalue
            
        statistics = ['Original value','CVAE result','MCMC result','KS test for normality','KS p-value for normality']
        print("&" + "&".join(statistics) + '\\\\\\hline')
        for which in range(len(names)):
            print( names[which] + "&" + "{:.4f}&{:.4f} &{:.4f} $\\pm$ {:.4f}&{:.4f}&{:.3E}\\\\".\
                   format(constant_values[which],\
                          constant_values[which]*perturbation_factors_NN[5+which],\
                          mu[which] ,\
                          sigma[which] ,\
                          ks_value[which] ,\
                          ks_pvalue[which]))


def mcmc(output_path = MODEL_HOME + '/experiments/mcmc', perturbation_factors_path = MODEL_HOME + '/settings/perturbation_factors/perturbation_factors_history_AM_test.npy',constant_path1 = MODEL_HOME + '/settings/cte_lambda.csv',constant_path2=MODEL_HOME + '/settings/cte.csv',chla_hat_path = MODEL_HOME+'/settings/reproduce/results_AM',rank=0,nranks=1):
    my_device = 'cpu'
    precision = torch.float64
    perturbation_factors = torch.tensor(np.load(perturbation_factors_path)[-1]).to(precision)
    constant = read_constants(file1=constant_path1,file2=constant_path2,my_device = my_device,precision=precision)

    chla_hat = np.load(chla_hat_path+'/X_hat.npy')
    
    mu_z = torch.tensor(chla_hat[:,::2],dtype=precision)
    train_labels = np.load(MODEL_HOME+'/settings/npy_data/train_labels.npy')
    train_labels= np.sort(train_labels)
    mu_z = mu_z[train_labels]
    for j in range(rank,40,nranks):
        torch.manual_seed(j)
        saving_mcmc((j,perturbation_factors,constant,mu_z.unsqueeze(1),output_path))
    #list(map(saving_mcmc,[(j,perturbation_factors,constant,mu_z.unsqueeze(1),output_path) for j in range(40)]))
                

def analize_mcmc_result(perturbation_factors_path=MODEL_HOME + '/settings/reproduce/perturbation_factors',results_path = MODEL_HOME+'/settings/reproduce',constant_path1 = MODEL_HOME+'/settings',constant_path2 = MODEL_HOME+'/settings' ,mcmc_runs_path = MODEL_HOME + '/settings/reproduce/mcmc',plots_path = MODEL_HOME + '/settings/reproduce/plots'):
    
    data_dir = MODEL_HOME + '/settings/npy_data'

    my_device = 'cpu'
    precision = torch.float64
    
    indexes = customTensorData(data_path=data_dir,which='all',per_day = True,randomice=True,one_dimensional = False,seed = 1853,device=my_device).train_indexes
    #perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/settings/perturbation_factors/perturbation_factors_history_AM_test.npy')[-1]).to(torch.float32)
    perturbation_factors = torch.tensor(np.load(perturbation_factors_path + '/perturbation_factors_history_loss_normilized.npy')[-1]).to(precision)

    perturbation_factors_NN = torch.tensor(np.load(perturbation_factors_path + '/perturbation_factors_history_CVAE_chla_centered.npy')[-500:]).to(torch.float32).mean(axis=0)
    perturbation_factors_NN_std = torch.tensor(np.load(perturbation_factors_path + '/perturbation_factors_history_CVAE_chla_centered.npy')[-300:]).to(torch.float32).std(axis=0)

    chla_hat = torch.tensor(np.load(results_path + '/results_AM/X_hat.npy'),dtype=precision)[:,::2].unsqueeze(1)[indexes]

    constant = read_constants(file1=constant_path2 + '/cte_lambda.csv',file2=constant_path2 + '/cte.csv',my_device = my_device,dict=True,precision=precision)
    constant_values = np.array([constant['dCDOM'],constant['sCDOM'],5.33,0.45,constant['Theta_min'],constant['Theta_o'],constant['beta'],constant['sigma'],0.005])
   
    num_runs = 40
    mcmc_runs = np.empty((num_runs,3000,14))

    for i in range(num_runs):
        mcmc_runs[i] = np.load(mcmc_runs_path + '/run_' + str(i)+'.npy')
        
    correlation_matrix(num_runs,mcmc_runs,plot=False,table=True)

    
    #mcmc_runs = mcmc_runs[:,2000:,:]
    perturbation_factors_mcmc = np.reshape(mcmc_runs[:,1500::499,:],(mcmc_runs[:,1500::499,:].shape[0]*mcmc_runs[:,1500::499,:].shape[1],14))
    
    np.save(perturbation_factors_path + '/perturbation_factors_mcmc_std.npy',perturbation_factors_mcmc.std(axis=0))
    np.save( perturbation_factors_path + '/perturbation_factors_mcmc_mean.npy',perturbation_factors_mcmc.mean(axis=0))
    perturbation_factors_mcmc = perturbation_factors_mcmc.mean(axis=0)
    


    #correlation_matrix(num_runs,mcmc_runs,plot=False,table=True)

    mcmc_runs = mcmc_runs[:,:,5:] * constant_values

    mcmc_runs_mean = np.mean(mcmc_runs,axis=0)
    mcmc_percentile_2_5 = np.percentile(mcmc_runs,2.5,axis=0)
    
    mcmc_percentile_98_5 = np.percentile(mcmc_runs,98.5,axis=0)
    mcmc_percentile_16 = np.percentile(mcmc_runs,16,axis=0)
    mcmc_percentile_84 = np.percentile(mcmc_runs,84,axis=0)
    
    
    fig_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    names = ['$d_{\mathrm{CDOM}}$ [$\mathrm{m}^2(\mathrm{mgCDOM})^{-1}$]','$S_{\mathrm{CDOM}}$ [nm]','$Q_a$','$Q_b$',\
             '$\Theta^{\mathrm{min}}_{\mathrm{chla}}$ [$\mathrm{mgChla}\mathrm{(mgC)}^{-1}$]','$\Theta^{\mathrm{0}}_{\mathrm{chla}}$  [$\mathrm{mgChla}\mathrm{(mgC)}^{-1}$]',\
             '$\\beta$ [$\mathrm{mmol}\mathrm{m}^{-2}\mathrm{s}^{-1}$]','$\sigma$  [$\mathrm{mmol}\mathrm{m}^{-2}\mathrm{s}^{-1}$]','$b_{r,\mathrm{NAP}}$']
    which = 4
    fig,axs = plt.subplots(ncols=2,nrows=1,layout='constrained',width_ratios=[3/4,1/4])

    axs[0].axhline( y = constant_values[which],linestyle='--',label='original value',color='black')
    print(mcmc_runs_mean)

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

    data = mcmc_runs[:,2000:,which]
    data = data[:,::280].flatten()
    data = data
    
    
    n,bins,patches = axs[1].hist(data,orientation='horizontal',bins=40,edgecolor='black',color='#377eb8',alpha=0.6,density=True)
    (mu, sigma) = scipy.stats.norm.fit(data)
    axs[1].plot(scipy.stats.norm.pdf(bins,loc=mu,scale=sigma),bins , linewidth=2,color='black',label='normal distribution\nmean: {:.3f}\nstd: {:.3f}'.format(mu,sigma))
    axs[1].legend()
    axs[1].text(-0.1,1.05,'(b)',transform = axs[1].transAxes,fontsize=20)
    axs[1].set_xlabel('Probability density')

    axs[1].set_yticks([])
    axs[0].set_ylim(*axs[1].get_ylim())
    plt.savefig(plots_path + '/mcmc_example.pdf')
    plt.close()

    mcmc_runs = mcmc_runs[:,2000:,:]

    print(mcmc_runs.shape)


    parameters_statistics(num_runs,mcmc_runs,names,correlation_lenght=280,plot=True,table=True,constant_values=constant_values,perturbation_factors = perturbation_factors_mcmc,perturbation_factors_NN = perturbation_factors_NN,fig_labels=fig_labels,perturbation_factors_NN_std = perturbation_factors_NN_std,plots_path = plots_path)
    

if __name__ == '__main__':
    torch.set_num_threads(1)
    
    #profiling_likelihood(perturbation_factors_path = MODEL_HOME + '/settings/reproduce_dukiewicz/perturbation_factors/perturbation_factors_mcmc_mean.npy',constant_path1=MODEL_HOME+'/settings/cte_lambda_dukiewicz/cte_lambda.csv',constant_path2=MODEL_HOME+'/settings/cte.csv',chla_hat_path = MODEL_HOME+'/settings/reproduce_dukiewicz/results_AM',output_path=MODEL_HOME + '/settings/reproduce_dukiewicz/plots/likelihood_profile.npy')
    #sys.exit()
    
    #sys.exit()
    #mcmc(output_path = MODEL_HOME + '/settings/reproduce/mcmc/', perturbation_factors_path = MODEL_HOME + '/settings/reproduce/perturbation_factors/perturbation_factors_history_loss_normilized.npy',constant_path1 = MODEL_HOME + '/settings/cte_lambda.csv',constant_path2=MODEL_HOME + '/settings/cte.csv',chla_hat_path = MODEL_HOME+'/settings/reproduce/results_AM')
    comm = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nranks = comm.size
    
    #mcmc(output_path = MODEL_HOME + '/settings/reproduce_dukiewicz/mcmc/', perturbation_factors_path = MODEL_HOME + '/settings/reproduce_dukiewicz/perturbation_factors/perturbation_factors_history_loss_normilized.npy',constant_path1 = MODEL_HOME + '/settings/cte_lambda_dukiewicz/cte_lambda.csv',constant_path2=MODEL_HOME + '/settings/cte.csv',chla_hat_path = MODEL_HOME+'/settings/reproduce_dukiewicz/results_AM',rank=rank,nranks=nranks)
    comm.Barrier()
    if rank == 0:

        analize_mcmc_result(perturbation_factors_path=MODEL_HOME + '/settings/reproduce_dukiewicz/perturbation_factors',results_path = MODEL_HOME+'/settings/reproduce_dukiewicz',constant_path1 = MODEL_HOME+'/settings/cte_lambda_dukiewicz',constant_path2 = MODEL_HOME+'/settings' ,mcmc_runs_path = MODEL_HOME + '/settings/reproduce_dukiewicz/mcmc',plots_path = MODEL_HOME + '/settings/reproduce_dukiewicz/plots')
    #analize_mcmc_result(perturbation_factors_path=MODEL_HOME + '/settings/reproduce/perturbation_factors',results_path = MODEL_HOME+'/settings/reproduce',constant_path1 = MODEL_HOME+'/settings',constant_path2 = MODEL_HOME+'/settings' ,mcmc_runs_path = MODEL_HOME + '/settings/reproduce/mcmc',plots_path = MODEL_HOME + '/settings/reproduce/plots')
