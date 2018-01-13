# import sys
# sys.modules[__name__].__dict__.clear()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import scipy.optimize 
from scipy.integrate import odeint

from nonlinearcoefficient import *
from CreateBeam import *
from RK4_NLO_1Dt import *

c= 3e8 #speed of light [m/s]
eps0 = 8.85e-12 #vacuum permittivity [F/m]

step_z = 1e-6 #step of simulation [m]


Nt = 2**10 #number of points in time domain
resol_t = 1e-15 #time resolution [s]


time_vect = resol_t*np.linspace(-Nt/2,Nt/2-1,Nt)
nu_vect = 1/2/resol_t*np.linspace(-1,1,Nt+1) 
#pulsation vector not shifted [rad/s]
nu_vect = nu_vect[:-1]

derivative_step_omega = 1e12 #step in omega to perform numerical derivative [rad/s]

## Physical information
#--------Crystal info--------
param_crystal = {
	'crystal' : 'BBO', #string for the nonlinear crystal	
	'theta_crystal' : 30.86, #cut angle of the crystal
	'phi_crystal' : 0, #cut angle of the crystal
	'L' : 1.5e-3 #Length of crystal
}
Type = 'FSS' #type of the interaction {'FSS','FSF','FFS'} (fast=F,slow=S) (i.e type I = FSS)

d = nonlinearCoefficientBBO()*1e-12 #non linear coefficient matrix [m/V]

param_pump = {
	'Fluence': 24.5, #Fluence of the signal [J/m^2]
	'Lambda': 800e-9, #signal's central wavelength [m]
	'FWHMt': 40e-15, # signal's FWHM in Intensity in time domain [s]
	'Phi1': 0e-15, # signal first order spectral phase (linked to group delay) [s]
	'Phi2': 0e-30, # signal second order spectral phase (linked to linear frequency chirp) [s^2]
	'Phi3': 0e-45, # signal third order spectral phase [s^3]
}
param_pump['omega'] = 2*np.pi*c/param_pump['Lambda'] #signal central pulsation [rad/s]

amplitude_pump, index_ref_pump, Efield_pump =\
create_beam(param_pump, param_crystal, Type[0], nu_vect)

k0_pump, k1_pump, k2_pump = Get_K0_k1_k2(param_pump, index_ref_pump, param_crystal, Type[0], derivative_step_omega)

param_signal = {
	'Fluence': 0.5,
	'Lambda': 1800e-9,
	'FWHMt': 40e-15,
	'Phi1': 0e-15,
	'Phi2': 0e-30,
	'Phi3': 0e-45,
}
param_signal['omega'] = 2*np.pi*c/param_signal['Lambda'] #signal central pulsation [rad/s]

amplitude_signal, index_ref_signal, Efield_signal = create_beam(param_signal, param_crystal, Type[1], nu_vect)
k0_signal, k1_signal, k2_signal = Get_K0_k1_k2(param_signal, index_ref_signal, param_crystal, Type[1], derivative_step_omega)

param_idler = {
	'Fluence': 0,
	'Lambda': (1/param_pump['Lambda'] - 1/param_signal['Lambda'])**(-1),
	'FWHMt': 40e-15,
	'Phi1': 0e-15,
	'Phi2': 0e-30,
	'Phi3': 0e-45,
}

param_idler['omega'] = 2*np.pi*c/param_idler['Lambda'] #signal central pulsation [rad/s]

amplitude_idler, index_ref_idler, Efield_idler = create_beam(param_idler, param_crystal, Type[2], nu_vect)
k0_idler, k1_idler, k2_idler = Get_K0_k1_k2(param_idler, index_ref_idler, param_crystal, Type[2], derivative_step_omega)

Efield_combi = np.array([\
					Efield_pump[0]*Efield_signal[0],
					Efield_pump[1]*Efield_signal[1],
					Efield_pump[2]*Efield_signal[2],
					Efield_pump[1]*Efield_signal[2] + Efield_pump[2]*Efield_signal[1],
					Efield_pump[0]*Efield_signal[2] + Efield_pump[2]*Efield_signal[0],
					Efield_pump[0]*Efield_signal[1] + Efield_pump[1]*Efield_signal[0] 
    ]) 

# vector needed to multiply with nonlinear matrix (see p41 Boyd)

P = d.dot(Efield_combi) #polarisation of idler
P = np.transpose(P).dot(Efield_idler/MyNorm(Efield_idler)) #projection of polarisation on Electric field of idler
deff = MyNorm(P)/MyNorm(Efield_pump)/MyNorm(Efield_signal) #scalar value of the nonlinear matrix [m/V]

deltak = k0_pump - k0_signal - k0_idler

## Simulation

Nz = int(np.round(param_crystal['L']/step_z))
z = 0

## NLO PROPAGATION in crystal

for i in range(Nz):
    amplitude_pump, amplitude_signal, amplitude_idler = RK4_NLO_1Dt(amplitude_pump,amplitude_signal,amplitude_idler,\
        param_pump['omega'], param_signal['omega'], param_idler['omega'],index_ref_pump, index_ref_signal, index_ref_idler,\
        k1_pump,k1_signal,k1_idler,k2_pump,k2_signal,k2_idler,\
        2*np.pi*nu_vect,deff,deltak,z,step_z)
    z = z+step_z 

# Time domain

field_pump = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(amplitude_pump)))
field_signal = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(amplitude_signal)))
field_idler = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(amplitude_idler)))

# plottons les resultats dans le domaine temporel
fig1 = plt.figure(1)
plt.clf()
Pump, = plt.plot(time_vect,np.transpose(np.abs(field_pump)**2/np.max(np.abs(field_pump)**2)))
Signal, = plt.plot(time_vect,np.transpose(np.abs(field_signal)**2/np.max(np.abs(field_signal)**2)))
Idler, = plt.plot(time_vect,np.transpose(np.abs(field_idler)**2/np.max(np.abs(field_idler)**2)))
plt.legend([Pump, Signal, Idler], ['pump','signal','idler'])
plt.xlim([-2.5e-13, 2.5e-13])
plt.show()

## On verifie la condition de conservation de l'energie

end_fluence_pump = c*eps0*index_ref_pump/2*np.trapz(np.abs(amplitude_pump)**2, x = nu_vect)
end_fluence_signal = c*eps0*index_ref_signal/2*np.trapz(np.abs(amplitude_signal)**2, x = nu_vect)
end_fluence_idler = c*eps0*index_ref_idler/2*np.trapz(np.abs(amplitude_idler)**2, x = nu_vect)

end_energy = end_fluence_pump + end_fluence_signal + end_fluence_idler