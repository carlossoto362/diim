
#First, we load the necessary python modules, and data requiered.
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy import stats
from datetime import datetime,timedelta
from torch.utils.data import DataLoader,random_split
import warnings
import os
import sys

###############################################################################
#We would also like to import a module to read the data we are going to use. The library is called read_data_module and is stored in the diimpy library. After importing it, let's select the test data to work with. We used the train data for work, while test data was left only for testing the results. In this module, let's use the test data since is a smaller set. The seed used for selecting the data was 1853.
###############################################################################

my_device = 'cpu'
if 'DIIM_PATH' in os.environ:
    HOME_PATH = MODEL_HOME = os.environ["DIIM_PATH"] #most of the codes have this lines, make sure DIIM_PATH is in your path. 
else:

    print("Missing local variable DIIM_PATH. \nPlease add it with '$:export DIIM_PATH=path/to/diim/model'.")
    sys.exit()

import diimpy.read_data_module as rdm


data = rdm.customTensorData(data_path=MODEL_HOME + '/settings/npy_data',which='test',per_day = True,randomice=True,seed=1853,normilized_NN='scaling')
constant = rdm.read_constants(file1=MODEL_HOME + '/settings/cte_lambda.csv',file2=MODEL_HOME + '/settings/cte.csv',my_device = my_device)

print('printing column names of OASIM model data: ',data.x_column_names)
print('printing column names of satellite data',data.y_column_names)
print('Printing dictionary with the constants of the model: ',constant)

###############################################################################
#We would also like to load the forward model, also stored in the diim library.
###############################################################################

import diimpy.Forward_module as fm

batch_size = data.len_data

model = fm.Forward_Model(num_days=1).to(my_device)
X,Y = data.__getitem__(0)

def return_date(day):
  return datetime(year=2000,month=1,day=1) + timedelta(days = int(data.dates[0]))

print("printing $E_{dir}, E_{dif}, \lambda, zenith, PAR$ for day" + " {}".format( return_date(data.dates[0]).strftime("%d/%m/%Y") ))
print(X)

print("printing $R_{RS,412.5},R_{RS,442.5},R_{RS,490},R_{RS,510},R_{RS,555}$ for day " + "{}".format( return_date(data.dates[0]).strftime("%d/%m/%Y") ))
print(Y)

###############################################################################
#Now, the RRS depends also in (chla,NAP,CDOM), which, for now, we don't know. Let's define a random tensor, and use it to evaluate the function. We also need to specify the perturbation factors, which are values that modify the existing parameters. If we set all of them to 1, we would be using the parameters from the literature.
###############################################################################

perturbation_factors = torch.ones(14, dtype=torch.float32)

chla = torch.rand((1,1,3))
RRS_OBS = Y
RRS_PRED = model(X.unsqueeze(0),perturbation_factors_ = perturbation_factors,constant=constant)
print(RRS_OBS,RRS_PRED)

###############################################################################
#The values are not close to the measured ones, because we used random numbers as constituents. In our work, we used a Bayesian framework to find the optical constituents that maximized the posterior distribution, p(z|y,x), where z are the optical constituents, chla,NAP,CDOM, y the satellite data, RRS, and x the OASIM model data, Edir,Edif,λ,zenith,PAR. Computationally, we defined a loss function, "RRS_loss" equal to −2log(p(z|y,x)), part of the operational method, and minimized the loss function using a fix set of parameters.

#To start, we defined a train loop, which using the Adam altorithm, minimizes the loss function "RRS_loss". The train loop is in the module bayes_inversion.py
###############################################################################

import diimpy.bayesian_inversion as bayes
#print(help(bayes.train_loop))


###############################################################################
#We also want to optimize the forward model by optimizing his parameters, such that the inversion returns values close to observe data. For this, we also defined a loss function, OBS_loss, and minimize it. Since for each iteration of the minimization of OBS_loss, involves a train loop minimizing RRS_loss, we started by finding an approximation of the minimum of both by using alternate minimization. The function that does the alternate minimization is track_parameters, also in the module bayesian_inversion.
###############################################################################

perturbation_factors = bayes.track_parameters(data_path = MODEL_HOME + '/settings/npy_data',output_path = MODEL_HOME + '/plot_data/perturbation_factors',iterations=100,save=False,which = 'test', seed = 1853,name='history.npy')
print(perturbation_factors[-1])

###############################################################################
#The Alternate Minimization manage to minimize both loss functions, RRS_loss and OBS_loss, but doesn't help us if we want to quantify the uncertainty. For that, we used the Metropolis algorithm to sample for the posterior, using e−(0.5)OBSloss as the model for the likelihood. Again, our likelihood depends in chla, nap and cdom, which are obtained by minimizing RRS_loss, so instead, we approximate the result by only performing a limited set of iterations. The final result is in the module sensitivity_analisys, together with the sensitivity analysis. Here we can save some time and load the MCMC runs, from the folder mcmc/ or mcmc/runs2/.
###############################################################################

num_runs = 40
mcmc_runs = np.empty((num_runs,3000,14))

for i in range(num_runs):
        mcmc_runs[i] = np.load(MODEL_HOME + '/experiments/mcmc/run_' + str(i)+'.npy')

        
###############################################################################        
#In addition, we also found values for the parameters using a framework called SGVB, which for now, I'm only going to mention that involves training a Neural Network. So, we can compare the mcmc parameters, the result of the Alternate Minimization (stored in /settings/perturbation_factors) and the results using a neural network. 
###############################################################################

perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/settings/perturbation_factors/perturbation_factors_history_AM_test.npy')[-1]).to(torch.float32)
perturbation_factors_NN = torch.tensor(np.load(MODEL_HOME + '/settings/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy')[-300:]).to(torch.float32).mean(axis=0)

###############################################################################
#Using this two results, with the mcmc runs, we can make some plots to compare each other. For example:
###############################################################################

import matplotlib.pyplot as plt
constant_values = np.array([constant['dCDOM'],constant['sCDOM'],5.33,0.45,constant['Theta_min'],constant['Theta_o'],constant['beta'],constant['sigma'],0.005])


#mcmc_runs = mcmc_runs[:,2000:,:]
mcmc_runs_mean, mcmc_runs_std = np.empty((14)),np.empty((14))


#correlation_matrix(num_runs,mcmc_runs,plot=False,table=False)

mcmc_runs = mcmc_runs[:,:,5:] * constant_values

mcmc_runs_mean = np.mean(mcmc_runs,axis=0)
mcmc_percentile_2_5 = np.percentile(mcmc_runs,2.5,axis=0)

mcmc_percentile_98_5 = np.percentile(mcmc_runs,98.5,axis=0)
mcmc_percentile_16 = np.percentile(mcmc_runs,16,axis=0)
mcmc_percentile_84 = np.percentile(mcmc_runs,84,axis=0)


fig_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
names = ['$d_{\\text{CDOM}}$ [$\\text{m}^2(\\text{mgCDOM})^{-1}$]','$S_{\\text{CDOM}}$ [nm]','$Q_a$','$Q_b$',\
             '$\Theta^{\\text{min}}_{\\text{chla}}$ [$\\text{mgChla}\\text{(mgC)}^{-1}$]','$\Theta^{\\text{0}}_{\\text{chla}}$  [$\\text{mgChla}\\text{(mgC)}^{-1}$]',\
             '$\\beta$ [$\\text{mmol}\\text{m}^{-2}\\text{s}^{-1}$]','$\sigma$  [$\\text{mmol}\\text{m}^{-2}\\text{s}^{-1}$]','$b_{b,\\text{NAP}}$']
which = 3
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
plt.show()


