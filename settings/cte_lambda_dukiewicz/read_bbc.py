import numpy as np
import scipy
import sys

file_ = open('acbc25b_carbon.dat','r')
for i in range(7):
	file_.readline()
Diatoms = {'lam':[],'a':[],'b':[],'bb':[],'name':'Diatoms'}
for i in range(19):
        line = file_.readline().split('   ')
        Diatoms['lam'].append(int(line[0]))
        Diatoms['a'].append(float(line[1]))
        Diatoms['b'].append(float(line[3]))
        Diatoms['bb'].append(float(line[4][:-1]))

Flagellates = {'lam':[],'a':[],'b':[],'bb':[],'name':'Flagellates'}
file_.readline()
for i in range(19):
        line = file_.readline().split('   ')
        Flagellates['lam'].append(int(line[0]))
        Flagellates['a'].append(float(line[1]))
        Flagellates['b'].append(float(line[3]))
        Flagellates['bb'].append(float(line[4][:-1]))
Pico = {'lam':[],'a':[],'b':[],'bb':[],'name':'Pico'}
file_.readline()
for i in range(19):
        line = file_.readline().split('   ')
        Pico['lam'].append(int(line[0]))
        Pico['a'].append(float(line[1]))
        Pico['b'].append(float(line[3]))
        Pico['bb'].append(float(line[4][:-1]))
Dino = {'lam':[],'a':[],'b':[],'bb':[],'name':'Dino'}
file_.readline()
for i in range(19):
        line = file_.readline().split('   ')
        Dino['lam'].append(int(line[0]))
        Dino['a'].append(float(line[1]))
        Dino['b'].append(float(line[3]))
        Dino['bb'].append(float(line[4][:-1]))
file_.close()

phys = [Diatoms,Flagellates,Pico,Dino]

for phy in phys:
        phy['a'] = np.array(phy['a'])
        phy['b'] = np.array(phy['b'])
        phy['bb'] = np.array(phy['bb'])
        phy['lam'] = np.array(phy['lam'])

lambdas = np.array([412.5,442.5,490,510,555])
a_phy = np.zeros(5)
b_phy = np.zeros(5)
bb_phy = np.zeros(5)
for phy in phys:
        a_phy += np.interp(lambdas, phy['lam'], phy['a'])
        b_phy += np.interp(lambdas, phy['lam'], phy['b'])
        bb_phy += np.interp(lambdas, phy['lam'], phy['bb'])
a_phy = a_phy/5
b_phy = b_phy/5
bb_phy = (bb_phy/5)*b_phy
a_w = np.array([0.00271,0.00574,0.0146,0.033,0.06098])
b_w = np.array([0.00535,0.00437,0.00284,0.00247,0.00167])
bb_w = np.array([0.002674,0.002184,0.001421,0.001234,0.000836])
print('lambda,absortion_w,scattering_w,backscattering_w,absortion_PH,scattering_PH,backscattering_PH')
for i in range(5):
        #print('{:.1f},{:.5f},{:.5f},{:.6f},{:.5f},{:.5f},{:.2E}'.format(lambdas[i],a_w[i],b_w[i],bb_w[i],a_phy[i],b_phy[i],bb_phy[i]))
        print("{:.1f}&{:.5f}&{:.5f}&{:.6f}&{:.5f}&{:.5f}&{:.2E}\\\\ ".format(lambdas[i],a_w[i],b_w[i],bb_w[i],a_phy[i],b_phy[i],bb_phy[i]))
        



