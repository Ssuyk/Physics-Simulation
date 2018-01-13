# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:52:17 2017

@author: Vincent
"""

import numpy as np
import matplotlib.pyplot as plt


def Sellmeier_FS(Lambda):
    #Lambda in microns, T in celsius
#    KOLEV 2006/bruner2003
    A = 1
    B = 0.6961663
    C = 0.0684043 
    D = 0.4079426
    E = 0.1162414
    F = 0.8974794
    G = 9.896161
    return np.sqrt(A+B*Lambda**2/(Lambda**2-C**2)+D*Lambda**2/(Lambda**2-E**2)+F*Lambda**2/(Lambda**2-G**2))

def Sellmeier_BK7(Lambda):
    #Lambda in microns, T in celsius
#    KOLEV 2006/bruner2003
    A = 1
    B = 1.03961212
    C = 0.00600069867 
    D = 0.231792344
    E = 0.0200179144
    F = 1.01046945
    G = 103.560653
    return np.sqrt(A+B*Lambda**2/(Lambda**2-C)+D*Lambda**2/(Lambda**2-E)+F*Lambda**2/(Lambda**2-G))


def b_T(x):
    return 3.483933E-8*(x+273.15)**2

def c_T(x):
    return 1.607839E-8*(x+273.15)**2
#Sellmeier equation for PPLT (Type-0)
def Sellmeier_LITA_M(Lambda, T):
    #Lambda in microns, T in celsius
#    KOLEV 2006/bruner2003
    A = 4.502483
    B = 0.007294
    C = 0.185087 
    D = -0.02357
    E = 0.073423
    F = 0.199595
    G = 0.001
    H = 7.99724
    return np.sqrt(A+(B+b_T(T))/(Lambda**2-(C+c_T(T))**2)+E/(Lambda**2-F**2)+G/(Lambda**2-H**2)+D*Lambda**2)

def Sellmeier_LITA_M_freq(Omega, T):
    c = 3E14 #microns.s-1
    Lambda = 2*np.pi*c/Omega # lambda in micron
#    KOLEV 2006/bruner2003
    A = 4.502483
    B = 0.007294
    C = 0.185087 
    D = -0.02357
    E = 0.073423
    F = 0.199595
    G = 0.001
    H = 7.99724
    return np.sqrt(A+(B+b_T(T))/(Lambda**2-(C+c_T(T))**2)+E/(Lambda**2-F**2)+G/(Lambda**2-H**2)+D*Lambda**2)


def V_g(Lambda, n):
    c= 3e8 #m/s
    return c/(n[0:-1]-Lambda[0:-1]*np.diff(n)/np.mean(np.diff(Lambda)))

def V_g_omega(omega, n):
    c= 3e8 #m/s
    return c/(n[0:-1]-(2*np.pi*c/omega[0:-1])*np.diff(n)/np.mean(np.diff(2*np.pi*c/omega)))

def GVD(Lambda, V_g):
    c= 3e8
    return np.diff(1/V_g)/np.mean(np.diff(2*np.pi*c/(Lambda*1e-6)))

def GVD_omega(omega, V_g_omega):
#    c= 3e8
#    -omega[0:-2]**2/(2*np.pi*c)*
    return np.diff(1/V_g_omega)/np.mean(np.diff(omega))

#TEST
#Lambda = np.linspace(0.9,2.0,1024) #micron
#omega = 2*np.pi*3e14/Lambda
#
#T = 25 #celsius
#c = 3e8
##Delta_phi = (Sellmeier_LITA_M(Lambda,T))*1e-3*2*np.pi/(Lambda*1e-6)
##Delta_phi = (Sellmeier_LITA_M(Lambda,T)-1)*12e-3*2*np.pi/(Lambda*1e-6)
#
#plt.figure(2)
#plt.clf()
#plt.subplot(2,2,1)
#plt.grid() 
##plt.plot(omega, Sellmeier_LITA_M_freq(omega,T))
#plt.plot(Lambda, Sellmeier_FS(Lambda))
#plt.plot(Lambda, Sellmeier_BK7(Lambda))
#
#plt.ylabel('refraction index $n_e(\omega)$')
#plt.title(' $n_e(\lambda)$')
#
#plt.subplot(2,2,2)
#plt.grid()
##plt.plot(omega[0:-1]/1e15,c/V_g_omega(omega, Sellmeier_LITA_M_freq(omega,T)))
#plt.plot(Lambda[0:-1],c/V_g(Lambda, Sellmeier_FS(Lambda)))
#plt.plot(Lambda[0:-1],c/V_g(Lambda, Sellmeier_BK7(Lambda)))
#
#plt.ylabel('Group Velocity ($c/V_G$ units)')
#plt.title('$V_G(\lambda)$')
#
#plt.subplot(2,2,3)
#plt.grid()
##plt.plot(omega[0:-2]/1e15, GVD_omega(omega,V_g_omega(omega, Sellmeier_LITA_M_freq(omega,T)))*1e27)
#plt.plot(Lambda[0:-2], GVD(Lambda,V_g(Lambda, Sellmeier_FS(Lambda,)))*1e27)
#plt.plot(Lambda[0:-2], GVD(Lambda,V_g(Lambda, Sellmeier_BK7(Lambda,)))*1e27)
#
#plt.xlabel('$\lambda (\mu m)$')
#plt.ylabel('GVD($\lambda$) (fs$^2$/mm)')
#plt.title(' Group Velocity Dispersion')

#V_phi = 3e8/Sellmeier_LITA_M(Lambda,T)
#plt.subplot(2,2,4)
#plt.grid()
#delta_t, = plt.plot(Lambda, (Delta_phi)/(omega)*1e12-7, label = r'Dephasing - 7ps')
#vphi, = plt.plot(Lambda, 12e-3/(V_phi)*1e12-84, label=r'$V_\phi$ - 84ps')
#vg, = plt.plot(Lambda[0:-1], 12e-3/(V_g(Lambda, Sellmeier_LITA_M(Lambda,T)))*1e12-86, label=r'$V_g$-86ps')
#plt.legend(handles=[delta_t, vphi, vg])
#plt.xlabel('$\lambda (\mu m)$')
#plt.ylabel('$\Delta t$ (ps) ')
#plt.title(' Delay (ps) accumulated through L = 12mm of LITA')

#from matplotlib.backends.backend_pdf import PdfPages
#pp = PdfPages('figure.pdf')