# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:47:44 2017

@author: Boulot
"""


import sys
sys.path.insert(0, r'C:\Users\Vincent\Google Drive\Python\INRS\Propagation Faisceau\Sellmeier_PPLN')

#scientific + plot library
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap

#Personal Function
import refractive_index as RI
from CalculateQout import *
from ChangeDomain import *
import Q_matrix as QM
import basic_Kmatrix as BKM
import Fit_2D_fields as MyFit
from scipy.optimize import fmin, fmin_bfgs

Npts=2**10+1
c=2.997e8#m/s
xmax=0.050#meter
tmax=2000#fs
#tmax2 = 1200
x=np.linspace(-xmax,xmax,Npts)
t=np.linspace(-tmax,tmax,Npts)
#t2=np.linspace(-tmax2,tmax2,Npts)

Lambda=1.9e-6#m
f0=c/Lambda*1e-15
omega0=2*np.pi*f0

dt = 2*tmax/Npts
Omega=(2*np.pi/(dt)/(Npts-1))*np.linspace(-Npts/2,Npts/2-1,Npts)
Lambda_vec = 2*np.pi*c/(Omega*1e15)

delta_x=2*xmax/Npts
k=(2*np.pi/(delta_x)/(Npts-1))*np.linspace(-Npts/2,Npts/2-1,Npts)

[X,T]=np.meshgrid(x,t)
#[Xtemp,T2]=np.meshgrid(x,t)

[X2,OMEGA]=np.meshgrid(x,Omega)
[K,OMEGA2]=np.meshgrid(k,Omega)
#[K2,T2]=np.meshgrid(k,t)

#definition of the initial pulse

E0=1 #to be defined

#%spatially 
R=1e20#m
w=0.005#m

#%temporally

tFWHM=20;#fs
Tau=tFWHM/np.sqrt(np.log(2))/2# pulse width in fs (the formalism use the half width at 1/e2,
#% Enter the FWHM value and it will convert it to 1/e2)
Beta=-0.00000#temporal chirp

DeltaOmegaFWHM=1/Tau*2*np.sqrt(np.log(2))
DeltaOmegaFWat1_over_e2=4/Tau
DeltaLambdaFWHM=(DeltaOmegaFWHM/omega0*Lambda)*1e9#nm
DeltaLambdaFWat1_over_e2=(DeltaOmegaFWat1_over_e2/omega0*Lambda)*1e9#nm

#No pulse front tilt initially
PFT= 0 #m^-1.fs^-1
WFR = 0 #m^-1.fs^-1

#% Grating definition
lines=75#%53;%lignes/mm
Theta_i=np.arcsin(Lambda*lines*1e3/2)#Littrow
m=1#diffraction order
Theta_d=np.arcsin(np.sin(Theta_i)-m*Lambda*1e3*lines)#rad
#
deltaTheta_dout=0*np.pi/180
Theta_iout=np.arcsin(np.sin(Theta_d+deltaTheta_dout)+m*Lambda*1e3*lines)#rad
#
#%lens definition
f=0.30#m
zR=np.pi*w**2/(Lambda)
waistatfocus=(Lambda)*f/np.pi/w/np.sqrt(1+f**2/zR**2);

deltaL1=0;#m
deltaL2=0;#m
#propagation
L1=f+deltaL1;#m
L2=f+deltaL2;#m