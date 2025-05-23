
import numpy as np
import torch
import pandas as pd
import os
import scipy
from torch.utils.data import DataLoader
import time
import sys
import diimpy.Forward_module as fm
import diimpy.read_data_module as rdm
from torch.utils.data import DataLoader
from torch import nn
from diimpy.CVAE_final import NN_second_layer
from tqdm import tqdm

if 'DIIM_PATH' in os.environ:
    HOME_PATH = MODEL_HOME = os.environ["DIIM_PATH"]
else:
    
    print("Missing local variable DIIM_PATH. \nPlease add it with '$:export DIIM_PATH=path/to/diim/model'.")
    sys.exit()

@torch.jit.script
def get_jacobian_components(original_jacobian,len_: int,comp: int):
    new_jacobian = original_jacobian[0,:,0,:,:].reshape(comp,3).clone().unsqueeze(0)
    for i in range(1,len_):
        new_jacobian = torch.cat((new_jacobian,original_jacobian[i,:,i,:,:].reshape(comp,3).clone().unsqueeze(0)),0)
    return new_jacobian

def train_loop(data_i,model,loss_fn,optimizer,N,kind='all',num_days=1,my_device = 'cpu',constant = None,perturbation_factors_ = None, scheduler = True):
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

    
    ls_val=[]
    past_pred=torch.empty((N,num_days,3))

    time_init = time.time()

    criterium = 1
    criterium_2 = 0
    i=0
    if scheduler == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    X = data_i[0].to(my_device) 
    Y = data_i[1].to(my_device)
    s_a = (loss_fn.s_a)
    s_e = (loss_fn.s_e)
    s_e_inverse = (loss_fn.s_e_inverse)
    s_a_inverse = (loss_fn.s_a_inverse)

    dR = (1e-13)

    while (((criterium >dR ) & (i<N)) or ((criterium_2 < 100)&(i<N))):
        
        pred_ = model(X,constant = constant,perturbation_factors_ = perturbation_factors_)
        loss_ = loss_fn(Y,pred_,model.state_dict()['chparam'])
        loss_.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
                
        ls_val.append(loss_.item())
        past_pred[i] = model.state_dict()['chparam'][:,0,:]
        
        if i != 0:
            criterium = ls_val[-2] - ls_val[-1]
        if criterium <=dR:
            criterium_2+=1
        i+=1
        if scheduler == True:
            scheduler.step(loss_)
    last_i = i
    
    last_rrs = pred_.clone().detach()
    last_loss = loss_.clone().detach()

    del i,pred_,loss_,criterium,criterium_2,Y
    if kind == 'all':
        parameters_eval = list(model.parameters())[0].clone().detach()
        evaluate_model = fm.evaluate_model_class(model=model,X=X,constant = constant)

        K_x = torch.empty((len(parameters_eval),5,3),dtype=s_e_inverse.dtype)
        for i in range(len(parameters_eval)):
            evaluate_model.X = X[i].unsqueeze(0)
            K_x[i] = torch.autograd.functional.jacobian(evaluate_model.model_der,inputs=(torch.unsqueeze(parameters_eval[i],0)))[0,:,0,0,:]
        
        evaluate_model.X = X
        #I = torch.inverse( torch.transpose(K_x,1,2) @ ( s_e_inverse @ K_x ))
        S_hat = torch.inverse( torch.transpose(K_x,1,2) @ ( s_e_inverse @ K_x ) + s_a_inverse  )
        #I_n = (((I)**2).sum()).sqrt()/(len(parameters_eval))
        #S_hat_n = (((S_hat)**2).sum()).sqrt()/(len(parameters_eval))
        #print(I_n/S_hat_n )
        #sys.exit()
        
        
        X_hat = torch.empty((len(parameters_eval),6),dtype=s_e_inverse.dtype)
        X_hat[:,::2] = past_pred[last_i-1].clone().detach()
        X_hat[:,1::2] = torch.sqrt(torch.diagonal(S_hat,dim1=1,dim2=2).clone().detach())
        output = {'X_hat':X_hat,'RRS_hat':last_rrs,'loss':last_loss}
        del K_x,past_pred,last_rrs,X_hat

        kd_hat = torch.empty((len(parameters_eval),10),dtype=s_e_inverse.dtype)
        kd_values = evaluate_model.kd_der(parameters_eval)

        kd_derivative = torch.empty((len(parameters_eval),5,3),dtype=s_e_inverse.dtype)
        for i in range(len(parameters_eval)):
            evaluate_model.X = X[i].unsqueeze(0)
            kd_derivative[i] = torch.autograd.functional.jacobian(evaluate_model.kd_der,inputs=(torch.unsqueeze(parameters_eval[i],0)))[0,:,0,0,:]

        kd_delta = fm.error_propagation(kd_derivative,S_hat)
        kd_hat[:,::2] = kd_values.clone().detach()
        kd_hat[:,1::2] = torch.sqrt(kd_delta).clone().detach()
        output['kd_hat'] = kd_hat
        del kd_hat,kd_delta,kd_derivative,kd_values

        bbp_hat = torch.empty((len(parameters_eval),6),dtype=s_e_inverse.dtype)
        bbp_values = evaluate_model.bbp_der(parameters_eval)

        bbp_derivative = torch.empty((len(parameters_eval),3,3),dtype=s_e_inverse.dtype)
        for i in range(len(parameters_eval)):
            evaluate_model.X = X[i].unsqueeze(0)
            bbp_derivative[i] = torch.autograd.functional.jacobian(evaluate_model.bbp_der,inputs=(torch.unsqueeze(parameters_eval[i],0)))[0,:,0,0,:]
        
        bbp_delta = fm.error_propagation(bbp_derivative,S_hat)
        bbp_hat[:,::2] = bbp_values.clone().detach()
        bbp_hat[:,1::2] = torch.sqrt(bbp_delta).clone().detach()
        output['bbp_hat'] = bbp_hat
        del bbp_hat,bbp_delta,bbp_derivative,bbp_values,S_hat
        return output
    
    elif kind == 'parameter_estimation':
        
        evaluate_model = fm.evaluate_model_class(model=model,X=X,constant = constant)
        
        parameters_eval = list(model.parameters())[0]
        
        output = torch.empty((X.shape[0],9),dtype=s_e_inverse.dtype)
        output[:,0] = past_pred[last_i-1][:,0]
        output[:,1:6] = evaluate_model.kd_der(parameters_eval,perturbation_factors_ = perturbation_factors_)
        output[:,6:] = evaluate_model.bbp_der(parameters_eval,perturbation_factors_ = perturbation_factors_)
        del past_pred,X

        return output

    else:
        print("time for training...",time.time() - time_init)
        return last_rrs.clone().detach().numpy(),past_pred[last_i-1].clone().detach().numpy(),last_loss

    

class Parameter_Estimator(nn.Module):
    """
	Model that attempts to learn the perturbation factors. 
    """
    def __init__(self,precision=torch.float32):
        super().__init__()
        self.perturbation_factors = nn.Parameter(torch.ones(14, dtype=precision), requires_grad=True)
        self.precision = precision
        self.perturbation_factors_names = [
            '$\epsilon_{a,ph}$',
            '$\epsilon_{tangent,s,ph}$',
            '$\epsilon_{intercept,s,ph}$',
            '$\epsilon_{tangent,b,ph}$',
            '$\epsilon_{intercept,b,ph}$',
            '$\epsilon_{a,cdom}$',
            '$\epsilon_{exp,cdom}$',
            '$\epsilon_{q,1}$',
            '$\epsilon_{q,2}$',
            '$\epsilon_{theta,min}$',
            '$\epsilon_{theta,o}$',
            '$\epsilon_\\beta$',
            '$\epsilon_\sigma$',
            '$\epsilon_{b,nap}$',
    ]

    def forward(self,data,constant,model,loss,optimizer,num_iterations=100,batch_size=1,my_device='cpu',scheduler = False):

        return train_loop(data,model,loss,optimizer,num_iterations,kind='parameter_estimation',num_days = batch_size,\
                          my_device = my_device,constant = constant,perturbation_factors_ = self.perturbation_factors, scheduler = scheduler)


def initial_conditions_nn(F_model,data,constant,data_path,which,randomice = False,precision = torch.float32,my_device = 'cpu'):

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

    model_NN = NN_second_layer(output_layer_size_mean=3,number_hiden_layers_mean = number_hiden_layers_mean,\
                           dim_hiden_layers_mean = dim_hiden_layers_mean,alpha_mean=alpha_mean,dim_last_hiden_layer_mean = dim_last_hiden_layer_mean,\
                           number_hiden_layers_cov = number_hiden_layers_cov,\
                           dim_hiden_layers_cov = dim_hiden_layers_cov,alpha_cov=alpha_cov,dim_last_hiden_layer_cov = dim_last_hiden_layer_cov,x_mul=x_mul,x_add=x_add,\
                           y_mul=y_mul,y_add=y_add,constant = constant,model_dir = MODEL_HOME + '/settings/VAE_model').to(my_device)

    
    model_NN.load_state_dict(torch.load(MODEL_HOME + '/settings/VAE_model/model_second_part_chla_centered.pt'))
    model_NN.eval()

    data.one_dimensional = True 
    
    X,Y = next(iter(DataLoader(data, batch_size=data.len_data, shuffle=False)))

    z_hat,cov_z,mu_z,kd_hat,bbp_hat,rrs_hat = model_NN(X[:,:,list(range(15)) + [20,21]]) #we are working with \lambda as imput, but the NN dosent use it. 
    mu_z = mu_z* model_NN.y_mul[0] + model_NN.y_add[0]
        
    state_dict = F_model.state_dict()
    state_dict['chparam'] = mu_z.unsqueeze(1)
    F_model.load_state_dict(state_dict)
    
    data.one_dimensional = False


def track_parameters(data_path = MODEL_HOME + '/settings/npy_data',output_path = MODEL_HOME + '/settings/reproduce_dukiewicz/perturbation_factors',iterations=10,save = False, which = 'train', seed = 1853, name = 'perturbation_factors_history_new.npy' ,constant_path1 = MODEL_HOME + '/settings/cte_lambda_dukiewicz/cte_lambda.csv',constant_path2=MODEL_HOME + '/settings/cte.csv',steps_z=100,num_threads=1):
    """
    Performes Alternate Minimization between the active constituents and the parameters of the model.

        Parameters:

        data_path: path where the data is stored. 
        
        output_path: if save is True, then the final perturbation factors history is stored in output_path. 

        iterations: Number of iterations to perform alternate minimization. 
    
        save: If the perturbation factors history gets stored. 

        which: which set of data to used in the alternate minimization. Can be 'train', which is 90% of the data, or 'test' which is 10% of it. The data is selected randomly. 

        seed: For reproducibility, the seed used to select the data set. 

    """
    torch.set_num_threads(num_threads)
    global_init_time = time.time()
    precision = torch.float64
    my_device = 'cpu'
    data = rdm.customTensorData(data_path=data_path,which=which,per_day = False,randomice=True,seed=seed,device = my_device,precision=precision)

    dataloader = DataLoader(data, batch_size=data.len_data, shuffle=False)

    constant = rdm.read_constants(file1=constant_path1,file2=constant_path2,my_device = my_device,precision=precision)
    
    x_a = torch.zeros(3)
    s_a = torch.eye(3)*1.13
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf

    lr = 0.029853826189179603
    batch_size = int(data.len_data)
    model = fm.Forward_Model(num_days=batch_size,precision=precision).to(my_device)
    #initial_conditions_nn(model,data,constant,data_path,which='train',randomice = False) #carefull with this step
    loss = torch.jit.script(fm.RRS_loss(x_a,s_a,s_e,num_days=batch_size,my_device = my_device,precision=precision))
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)



    Parameter_lr = 0.001
    Parameter_model = Parameter_Estimator(precision=precision)
    
    (p_X,p_Y_nan) = next(iter(dataloader))
    p_X = (p_X[:,:,1:],p_X[:,:,0])
    p_Y = torch.masked_fill(p_Y_nan,torch.isnan(p_Y_nan),0)
    
    Parameter_loss = torch.jit.script(fm.OBS_loss(my_device=my_device,precision=precision,normalization_values = torch.tensor(data.y_std)))
    Parameter_optimizer = torch.optim.Adam(Parameter_model.parameters(),lr=Parameter_lr)
    
    p_ls = []

    p_past_parameters = torch.empty((iterations,14)).to(precision)

    scheduler_parameters = torch.optim.lr_scheduler.ReduceLROnPlateau(Parameter_optimizer, 'min')
    for i in tqdm(range(iterations)):

        parameters_iter_time = time.time()
        
        for param in model.parameters():
            param.requires_grad = True
        p_pred = Parameter_model(p_X,constant,model,loss,optimizer,batch_size = batch_size,num_iterations=steps_z, scheduler = False)
        p_pred = torch.masked_fill(p_pred,torch.isnan(p_Y_nan),0)
        for param in model.parameters():
            param.requires_grad = False

        p_loss = Parameter_loss(p_Y,p_pred,p_Y_nan)
        p_loss.backward()
        for param in Parameter_model.parameters():#seting nan gradients to cero (only move in direction where I have information)
            p_grad = param.grad
            p_grad[p_grad != p_grad ] = 0

        Parameter_optimizer.step()
        Parameter_optimizer.zero_grad()
        
        for index_,p in enumerate(Parameter_model.parameters()):
            p.data.clamp_(min=0.1,max=1.9)
                
        p_ls.append(p_loss.item())
        p_past_parameters[i] =  next(iter(Parameter_model.parameters()))

        if i > 1900:
            scheduler_parameters.step(p_loss)
        if i % 10 == 0:
            print(i,'loss: ',p_ls[-1],list(iter(Parameter_model.parameters())))

    print('Total time: ',time.time() - global_init_time )
    to_plot = p_past_parameters.clone().detach().numpy()

    if save == True:
        np.save(output_path + '/' + name,to_plot )
    else:
        return to_plot


def track_alphas(output_path = MODEL_HOME + '/experiments/results_bayes_lognormal_logparam/alphas',save=False,perturbation_factors_path=MODEL_HOME + '/settings/reproduce_dukiewicz/perturbation_factors/perturbation_factors_history_loss_normilized.npy'):
    if save == True:
        print(output_path )
    perturbation_path = perturbation_factors_path
    data_path = MODEL_HOME + '/settings/npy_data'
    precision = torch.float64

    data = rdm.customTensorData(data_path=data_path,which='all',per_day = True,randomice=False,precision=precision)
    #perturbation_factors = torch.tensor(np.load(perturbation_path + '/perturbation_factors_history_AM_test.npy')[-1]).to(precision)
    perturbation_factors = torch.tensor(np.load(perturbation_factors_path)[-1]).to(precision)
    my_device = 'cpu'
    constant = rdm.read_constants(file1=MODEL_HOME + '/settings/cte_lambda_dukiewicz/cte_lambda.csv',file2=MODEL_HOME + '/settings/cte.csv',my_device = my_device,precision=precision)
    if save == True:
        np.save(output_path + '/dates.npy',data.dates)
    lr = 0.029853826189179603
    s_a_ = torch.eye(3)
    x_a = torch.zeros(3)
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf
    batch_size = data.len_data
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    for alpha in tqdm(np.linspace(0.1,10,20)):
        s_a = s_a_*alpha
        model = fm.Forward_Model(num_days=batch_size,precision=precision).to(my_device)
        model.perturbation_factors = perturbation_factors
        #initial_conditions(data,batch_size,model) #carefull with this step
        loss = torch.jit.script(fm.RRS_loss(x_a,s_a,s_e,num_days=batch_size,my_device = my_device,precision=precision))
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
        output = train_loop(next(iter(dataloader)),model,loss,optimizer,4000,kind='all',\
                             num_days=batch_size,constant = constant,perturbation_factors_ = perturbation_factors, scheduler = True)
        
        if save == True:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            np.save(output_path + '/X_hat_'+str(alpha)+'.npy',output['X_hat'])
            np.save(output_path + '/kd_hat_'+str(alpha)+'.npy',output['kd_hat'])
            np.save(output_path + '/bbp_hat_'+str(alpha)+'.npy',output['bbp_hat'])
            np.save(output_path + '/RRS_hat_'+str(alpha)+'.npy',output['RRS_hat'])
        print(alpha,'done')


def metropolis_daily():

    precision = torch.float64
    data = rdm.customTensorData(data_path=MODEL_HOME + '/settings/npy_data',which='train',per_day = True,randomice=False,precision=precision)
    from datetime import datetime,timedelta
    perturbation_factors = torch.ones(14).to(precision)
    perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/settings/reproduce_dukiewicz/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy')[:300].mean(axis=0)).to(precision)
    my_device = 'cpu'
    constant = rdm.read_constants(file1=MODEL_HOME + '/settings/cte_lambda_dukiewicz/cte_lambda.csv',file2=MODEL_HOME + '/settings/cte.csv',my_device = my_device,precision=precision)

    lr = 0.029853826189179603
    x_a = torch.zeros(3)
    s_a_ = torch.eye(3)
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf
    batch_size = 1
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    s_a = s_a_*1.13

    model = fm.Forward_Model(num_days=batch_size,precision=precision).to(my_device)
    model.perturbation_factors = perturbation_factors
    loss = fm.RRS_loss(x_a,s_a,s_e,num_days=batch_size,my_device = my_device,precision=precision)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    data_i = next(iter(dataloader))
    Y = data_i[1].to(precision)
    output = train_loop(data_i,model,loss,optimizer,4000,kind='all',\
                        num_days=batch_size,constant = constant,perturbation_factors_ = perturbation_factors, scheduler = False)
    X_hat = output['X_hat'][0,::2]
    dX_hat = output['X_hat'][0,1::2]

    model.eval()

    #performing the metropolis-Hasting algorithm to plot the true probability distribution agains the estimated one
    def q_f(x,dx):
        return np.array([np.random.normal(scale=dx[0],loc=x[0]),np.random.normal(scale=dx[1],loc=x[1]),np.random.normal(scale=dx[2],loc=x[2])])
        
    def alpha_f(pix,piy):
        return np.min([piy/pix,1])
    
    def pi_f(x):
        pred = model(data_i[0].to(precision),constant = constant,perturbation_factors_ = perturbation_factors,parameters = torch.tensor([[x]]).to(precision))
        return loss(Y,pred,torch.tensor([[x]]).to(precision)).clone().detach().numpy()

    def sampling_f(x0,len_):
        xs = np.empty((len_,3))
        xs[0] = x0
        
        for i in tqdm(range(len_-1)):
            y = q_f(X_hat,dX_hat)
            alpha = alpha_f(pi_f(y),pi_f(xs[i]))
            random_n = np.random.rand()

            if random_n <= alpha:
                xs[i+1] = y
            else:
                xs[i+1] = xs[i]
        return xs
    
    xs = sampling_f(X_hat,50000)
    
    def autocorr(x,lags):
        corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
        return np.array(corr)

    #print(autocorr(xs[:,0],np.arange(1,10)))
    #print(autocorr(xs[:,1],np.arange(1,10)))
    #print(autocorr(xs[:,2],np.arange(1,10)))
    #########################################
    xs = xs[5000::50]
    
    import matplotlib.pyplot as plt
    
    fig,axs = plt.subplots(ncols=3,nrows=1,layout='constrained',width_ratios=[1,1,1],figsize = (15,5),tight_layout=True)
    labels = ['(a)','(b)','(c)']
    labels_h = ['log-posterior distribution of chlorophyll with','log-posterior distribution of NAP with','log-posterior distribution of CDOM with']
    xlims = [(0,3),(0,20),(0,5)]
    for i in range(3):
        
        n,bins,patches = axs[i].hist(xs[:,i],bins=60,edgecolor='black',color='#377eb8',alpha=0.8,density=True,label=labels_h[i]+'\nmean {:.2f} and std {:.2f}.'.format(np.mean(xs[:,i]),np.std(xs[:,i])))
        log_dist = scipy.stats.norm(loc=X_hat[i],scale=dX_hat[i])
        #log_dist = scipy.stats.norm(loc=np.mean(xs[:,i]),scale=np.std(xs[:,i]))
        print(scipy.stats.kstest(xs[:,i],log_dist.cdf))
        axs[i].plot(np.linspace(bins[0],bins[-1],1000),log_dist.pdf(np.linspace(bins[0],bins[-1],1000)) ,'--', linewidth=1.5,color='black',label='Normal approximate distribution with \nmean {:.2f} and std: {:.2f}'.format(log_dist.stats()[0],np.sqrt(log_dist.stats()[1])))
        axs[i].legend()
        axs[i].text(-0.1,1.05,labels[i],transform = axs[i].transAxes,fontsize=20)
        axs[i].set_ylabel('Probability density')
        axs[i].set_xlabel('Log density [$\log (mg$ $m^{-3})$]')
        #axs[i].set_xlim(*xlims[i])
        #axs[1].set_yticks([])
    fig.tight_layout()
    plt.savefig('/g100_work/OGS23_PRACE_IT/csoto/DIIM/settings/reproduce_dukiewicz/plots/one_day_inversion.pdf')

def run_save_result(num_threads=1,data_path = MODEL_HOME + '/settings/npy_data',precision=torch.float64,perturbation_factors_file = MODEL_HOME + '/settings/reproduce_dukiewicz/perturbation_factors/perturbation_factors_mcmc_mean.npy',unperturbed=False,my_device = 'cpu',constant_path1 = MODEL_HOME + '/settings/cte_lambda_dukiewicz/cte_lambda.csv',constant_path2=MODEL_HOME + '/settings/cte.csv',output_path = MODEL_HOME+'/settings/reproduce/results',report_loss = False,save_files = True,perturbation_history=True,mean_last=False):
    torch.set_num_threads(num_threads)
    data = rdm.customTensorData(data_path=data_path,which='all',per_day = True,randomice=False)
    if perturbation_history:
        if mean_last:
            perturbation_factors = torch.tensor(np.load(perturbation_factors_file)[300:].mean(axis=0)).to(precision)
        else:
            perturbation_factors = torch.tensor(np.load(perturbation_factors_file)[-1]).to(precision)
    else:
        perturbation_factors = torch.tensor(np.load(perturbation_factors_file)).to(precision)
    if unperturbed == True:
        perturbation_factors = torch.ones(14).to(precision)

    constant = rdm.read_constants(file1=constant_path1,file2=constant_path2,my_device = my_device,precision=precision)

    lr = 0.029853826189179603
    x_a = torch.zeros(3)
    s_a_ = torch.eye(3)
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf
    batch_size = data.len_data
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    s_a = s_a_*1.13

    model = fm.Forward_Model(num_days=batch_size,precision=precision).to(my_device)
    model.perturbation_factors = perturbation_factors
    loss = fm.RRS_loss(x_a,s_a,s_e,num_days=batch_size,my_device = my_device,precision=precision)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    output = train_loop(next(iter(dataloader)),model,loss,optimizer,4000,kind='all',\
                        num_days=batch_size,constant = constant,perturbation_factors_ = perturbation_factors, scheduler = True)

    if report_loss == True:
        print(output['loss'])
    if save_files == True:
        np.save(output_path + '/X_hat.npy',output['X_hat'])
        np.save(output_path + '/kd_hat.npy',output['kd_hat'])
        np.save(output_path + '/bbp_hat.npy',output['bbp_hat'])
        np.save(output_path + '/RRS_hat.npy',output['RRS_hat'])
        np.save(output_path + '/dates.npy',data.dates)


if __name__ == '__main__':

    #track_parameters(data_path = MODEL_HOME + '/settings/npy_data',output_path = MODEL_HOME + '/settings/perturbation_factors',iterations=1000,save=True )
    #track_parameters(data_path = MODEL_HOME + '/settings/npy_data',output_path = MODEL_HOME + '/settings/reproduce/perturbation_factors',iterations=1000,save=True,name='perturbation_factors_history_loss_normilized.npy' )
    
    #track_alphas(output_path = MODEL_HOME + '/experiments/results_bayes_lognormal_logparam/alphas',save=False)
    #track_alphas(output_path = MODEL_HOME + '/settings/reproduce/alphas',save=True)
    
    #metropolis_daily()

    track_parameters(data_path = MODEL_HOME + '/settings/npy_data',output_path = MODEL_HOME + '/settings/reproduce/perturbation_factors',iterations=3000,save=True,name='perturbation_factors_history_loss_normilized.npy',constant_path1 = MODEL_HOME + '/settings/cte_lambda_dukiewicz/cte_lambda.csv' ,steps_z=20,num_threads=1)
    
    #run_save_result(num_threads=1,data_path = MODEL_HOME + '/settings/npy_data',precision=torch.float64,perturbation_factors_file = MODEL_HOME + '/settings/reproduce_dukiewicz/perturbation_factors/perturbation_factors_history_loss_normilized.npy',unperturbed=False,my_device = 'cpu',constant_path1 = MODEL_HOME + '/settings/cte_lambda_dukiewicz/cte_lambda.csv',constant_path2=MODEL_HOME + '/settings/cte.csv',output_path = MODEL_HOME+'/settings/reproduce_dukiewicz/results_AM',report_loss = True,save_files=True,perturbation_history=True)

    run_save_result(num_threads=1,data_path = MODEL_HOME + '/settings/npy_data',precision=torch.float64,perturbation_factors_file = MODEL_HOME + '/settings/reproduce_dukiewicz/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy',unperturbed=False,my_device = 'cpu',constant_path1 = MODEL_HOME + '/settings/cte_lambda_dukiewicz/cte_lambda.csv',constant_path2=MODEL_HOME + '/settings/cte.csv',output_path = MODEL_HOME+'/settings/reproduce_dukiewicz/results_lognormal_VAEparam',report_loss = True,save_files=True,perturbation_history=True,mean_last=True)
    
    #run_save_result(num_threads=1,data_path = MODEL_HOME + '/settings/npy_data',precision=torch.float64,perturbation_factors_file = MODEL_HOME + '/settings/reproduce_dukiewicz/perturbation_factors/perturbation_factors_mcmc_mean.npy',unperturbed=False,my_device = 'cpu',constant_path1 = MODEL_HOME + '/settings/cte_lambda_dukiewicz/cte_lambda.csv',constant_path2=MODEL_HOME + '/settings/cte.csv',output_path = MODEL_HOME+'/settings/reproduce_dukiewicz/results_lognormal_mcmc',report_loss = True,save_files=True,perturbation_history=False)
    
    #run_save_result(num_threads=1,data_path = MODEL_HOME + '/settings/npy_data',precision=torch.float64,perturbation_factors_file = MODEL_HOME + '/settings/reproduce/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy',unperturbed=False,my_device = 'cpu',constant_path1 = MODEL_HOME + '/settings/cte_lambda.csv',constant_path2=MODEL_HOME + '/settings/cte.csv',output_path = MODEL_HOME+'/settings/reproduce/results_lognormal_VAEparam',report_loss = True,save_files=True,perturbation_history=True)
    
    #run_save_result(num_threads=1,data_path = MODEL_HOME + '/settings/npy_data',precision=torch.float64,perturbation_factors_file = MODEL_HOME + '/settings/reproduce/perturbation_factors/perturbation_factors_mcmc_mean.npy',unperturbed=False,my_device = 'cpu',constant_path1 = MODEL_HOME + '/settings/cte_lambda.csv',constant_path2=MODEL_HOME + '/settings/cte.csv',output_path = MODEL_HOME+'/settings/reproduce/results_lognormal_mcmc',report_loss = True,save_files=True,perturbation_history=False)
    
