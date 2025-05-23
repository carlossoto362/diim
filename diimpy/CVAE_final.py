from torch import nn
import torch
import sys
import os
from diimpy.Forward_module import Forward_Model,kd,bbp


if 'DIIM_PATH' in os.environ:
    HOME_PATH = MODEL_HOME = os.environ["DIIM_PATH"]
else:
        
    print("Missing local variable DIIM_PATH. \nPlease add it with '$:export DIIM_PATH=path/to/diim'.")
    sys.exit()


class NN_first_layer(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_celu_stack = nn.Sequential(
            nn.Linear(17,22),
            nn.CELU(alpha=0.8978238833058),
            nn.Linear(22,9),
            nn.CELU(alpha=0.8978238833058)
        )

    def forward(self, x):
        x = self.linear_celu_stack(x)
        return x

class NN_second_layer(nn.Module):

    def __init__(self,my_device='cpu',chla_centered=True ,precision = torch.float32,model_dir = MODEL_HOME + '/settings/VAE_model',constant=None):
        super().__init__()

        self.chla_centered = chla_centered
        self.my_device = my_device
        self.precision = precision

        self.flatten = nn.Flatten()
        
        self.first_layer = NN_first_layer().to(self.my_device).to(self.precision)
        self.first_layer.load_state_dict(torch.load(model_dir+'/model_first_part.pt'))
        self.first_layer.eval()
        for param in self.first_layer.parameters():
            param.requires_grad = False

        alpha_mean = 1.3822406736258
        self.linear_celu_stack_mean = nn.Sequential( 
            nn.Linear(9,18),
            nn.CELU(alpha=alpha_mean),
            nn.Linear(18,18),
            nn.CELU(alpha=alpha_mean),
            nn.Linear(18,18),
            nn.CELU(alpha=alpha_mean),
            nn.Linear(18,19),
            nn.CELU(alpha=alpha_mean),
            nn.Linear(19,3),
            nn.CELU(alpha=alpha_mean)
        ).to(self.precision)

        alpha_cov = 0.7414694152899
        self.linear_celu_stack_cov = nn.Sequential(
            nn.Linear(9,13),
            nn.CELU(alpha=alpha_cov),
            nn.Linear(13,11),
            nn.CELU(alpha=alpha_cov),
            nn.Linear(11,9),
            nn.CELU(alpha=alpha_cov)
        ).to(self.precision)
        x_mul = [0.00553, 0.0049, 0.00319, 0.00185, 0.00154, 14.159258, 17.414488, 17.922438, 17.255323, 16.621885, 22.62492, 28.752264, 31.301714, 31.038338, 30.96801, 45.952972, 586.7632]
        y_mul = [4.9446263, 0.2707657, 0.36490566, 0.2853425, 0.2712304, 0.16722172, 0.004955111, 0.0035161567, 0.004354275]
        x_add = [0.00203, 0.00204, 0.0022, 0.00196, 0.00117, 2.5095496, 2.9216232, 2.914767, 2.7880442, 2.4963162, 0.002233, 0.0019174, 0.0017812, 0.0018124, 0.0016976, 20.611382, 34.561928]
        y_add = [-3.1879756, 0.022968043, 0.01729829, 0.018620161, 0.01796932, 0.057412438, 0.0001245, 0.00038759335, 7.9225e-05]
        
        self.x_mul = torch.tensor(x_mul).to(self.precision).to(self.my_device)
        self.y_mul = torch.tensor(y_mul).to(self.precision).to(self.my_device)
        self.x_add = torch.tensor(x_add).to(self.precision).to(self.my_device)
        self.y_add = torch.tensor(y_add).to(self.precision).to(self.my_device)

        self.Forward_Model = Forward_Model(learning_chla = False, learning_perturbation_factors = True,precision=self.precision)
        self.bbp = bbp
        self.kd = kd
        self.constant = constant

    def rearange_RRS(self,x):
        lambdas = torch.tensor([412.5,442.5,490.,510.,555.])
        x_ = x*self.x_mul + self.x_add
        output = torch.empty((len(x),5,5))
        output[:,:,0] = x_[:,0,5:10]
        output[:,:,1] = x_[:,0,10:15]
        output[:,:,2] = lambdas
        output[:,:,3] = x_[:,:,15]
        output[:,:,4] = x_[:,:,16]
        return output.to(self.precision).to(self.my_device)

    def forward(self, image):
        x = self.first_layer(image)
        mu_z = self.linear_celu_stack_mean(x).flatten(1)
        if self.chla_centered == True:
            mu_z += torch.column_stack((x[:,0,0],x[:,0,0],x[:,0,0])) 
        Cholesky_z = torch.tril(self.linear_celu_stack_cov(x).flatten(1).reshape((x.shape[0],3,3)))/10
        epsilon = torch.randn(torch.Size([x.shape[0],1,3]),generator=torch.Generator().manual_seed(0),dtype=self.precision).to(self.my_device)

        z_hat = mu_z + torch.transpose(Cholesky_z@torch.transpose(epsilon,dim0=1,dim1=2),dim0=1,dim1=2).flatten(1) 
        z_hat_inter = z_hat
        z_hat = (z_hat * self.y_mul[0] + self.y_add[0]).unsqueeze(1)
        image = self.rearange_RRS(image)
        
        rrs_ = self.Forward_Model(image,parameters = z_hat,constant = self.constant)
        rrs_ = (rrs_ - self.x_add[:5])/self.x_mul[:5]
 
        kd_ = self.kd(9.,image[:,:,0],image[:,:,1],image[:,:,2],image[:,:,3],image[:,:,4],torch.exp(z_hat[:,:,0]),torch.exp(z_hat[:,:,1]),torch.exp(z_hat[:,:,2]),self.Forward_Model.perturbation_factors,self.constant)
        kd_ = (kd_  - self.y_add[1:6])/self.y_mul[1:6]

        bbp_ = self.bbp(image[:,:,0],image[:,:,1],image[:,:,2],image[:,:,3],image[:,:,4],torch.exp(z_hat[:,:,0]),torch.exp(z_hat[:,:,1]),torch.exp(z_hat[:,:,2]),self.Forward_Model.perturbation_factors,self.constant)[:,[1,2,4]]
        bbp_ = (bbp_ - self.y_add[6:9])/self.y_mul[6:9]

        cov_z = torch.transpose(Cholesky_z,dim0=1,dim1=2) @ Cholesky_z

        return z_hat_inter,cov_z,mu_z,kd_,bbp_,rrs_

