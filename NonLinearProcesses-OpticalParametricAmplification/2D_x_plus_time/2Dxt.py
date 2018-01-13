# import sys
# sys.modules[__name__].__dict__.clear()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from numpy.fft import *

from nonlinearcoefficient import *
from refractiveindex import *
from Create_beam2D import *
from RK4_NLO_2Dxt import *

def cosd(Theta):
	return np.cos(Theta*np.pi/180)

def sind(Theta):
	return np.sin(Theta*np.pi/180)

def RotMat(fast_axis_orientation, theta_crystal, phi_crystal):
	return np.dot(np.array([[cosd(fast_axis_orientation), sind(fast_axis_orientation), 0], [-sind(fast_axis_orientation), cosd(fast_axis_orientation), 0], [0, 0, 1]])\
	, np.dot(np.array([ [cosd(theta_crystal), 0, -sind(theta_crystal)], [0, 1, 0], [sind(theta_crystal), 0, cosd(theta_crystal)] ]),\
	np.array([ [cosd(phi_crystal), sind(phi_crystal), 0], [-sind(phi_crystal), cosd(phi_crystal), 0], [0, 0, 1] ])))    

c= 3e8 #speed of light [m/s]
eps0 = 8.85e-12 #vacuum permittivity [F/m]

step_z = 10e-6 #step of simulation [m]

Nx = 2**8 #number of points in x
Nt = 2**10 #number of points in time domain
resol_x = 750e-6 #space resolution [m]
resol_t = 1e-15 #time resolution [s]

x_vect = resol_x*np.linspace(-Nx/2,Nx/2-1,Nx)
time_vect = resol_t*np.linspace(-Nt/2,Nt/2-1,Nt)

X,T = np.meshgrid(x_vect, time_vect)

kx_vect = 2*np.pi/2/resol_x*np.linspace(-1,1, Nx+1) #pulsation vector +1 points in x [rad/m]
kx_vect = kx_vect[:-1]

omega_vect = 2*np.pi/2/resol_t*np.linspace(-1,1,Nt+1) 
omega_vect = omega_vect[:-1]

[KX, Omega] = np.meshgrid(kx_vect, omega_vect)
max_error_ref_index = 1e-5; #maximum error tolerated on the value of the refractive index
derivative_step_omega = 1e12 #step in omega to perform numerical derivative [rad/s]

## Physical information
#--------Crystal info--------
param_crystal = {
	'crystal' : 'BBO', #string for the nonlinear crystal	
	'theta_crystal' : 30.86, #cut angle of the crystal
	'phi_crystal' : 0, #cut angle of the crystal
	'L' : 1.5e-3, #Length of crystal
	'fast_axis_orientation' : 0 #fast axis orientation compare to x axis laboratory frame

}
Type = 'FSF' #type of the interaction {'FSS','FSF','FFS'} (fast=F,slow=S) (i.e type I = FSS)

d = nonlinearCoefficientBBO()*1e-12 #non linear coefficient matrix [m/V]

rotation_matrix = RotMat(param_crystal['fast_axis_orientation'], param_crystal['theta_crystal'], param_crystal['phi_crystal'])
   # rotation matrix to pass from crystal frame to laboratory frame

Pump = {
	'Energy' : 2.26, #Energy per meter of the signal [J/m^2]
	'Lambda' : np.array([800e-9]), #signal's central wavelength [m]
	'FWHMt' : 40e-15, # signal's FWHM in Intensity in time domain [s]
	'Phi1' : 0e-15, # signal first order spectral phase (linked to group delay) [s]
	'Phi2' : 0e-30, # signal second order spectral phase (linked to linear frequency chirp) [s^2]
	'Phi3' : 0e-45, # signal third order spectral phase [s^3]
	'FWHMx' : 20e-3, # pump's beam diameter on the x axis (Intensity)[m]
	'x0' : 0e-3, # pump displacement on the x axis [m]
	'curvature_radiusx' : np.array([0e-3,inf]), # pump's x-radius of curavture center (x,z) in vaccum of the phase front, positive = defoc, negative = foc, inf = collimated [m]
	'Alpha' : 0, # pump's angle between the k vector and z axis (polar angle) positive angle for positive x [deg]
}

Pump['omega'] = 2*np.pi*c/Pump['Lambda'] #signal central pulsation [rad/s]
Pump['k0'], Pump['k_in_crystal'], Pump['k0_theta'], Pump['k0_phi'] = get_k0_k_ktheta_kphi(Pump['Alpha'], rotation_matrix)
Pump['index'], Pump['EField'] =  refractiveindex(Pump['Lambda'], Pump['k0_theta'][0],Pump['k0_theta'][0],param_crystal['crystal'],Type[0])
Pump['k0'] = 2*np.pi/Pump['Lambda']*Pump['k0']*Pump['index'] # pump's k vector in laboratory frame
Pump['index'] = Pump['index']*np.ones((1,Nx)) # initializing refractive index vector of pump

pulse_pump = create_beam2Dxt(Pump, X, Omega)


Signal = {
	'Energy' : 0.04, # Energy per meter of the signal [J/m^2]
	'Lambda' : np.array([1800e-9]), #signal's central wavelength [m]
	'FWHMt' : 50e-15, # signal's FWHM in Intensity in time domain [s]
	'Phi1' : -60e-15, # signal first order spectral phase (linked to group delay) [s]
	'Phi2' : 0e-30, # signal second order spectral phase (linked to linear frequency chirp) [s^2]
	'Phi3' : 0e-45, # signal third order spectral phase [s^3]
	'FWHMx' : 20e-3, # pump's beam diameter on the x axis (Intensity)[m]
	'x0' : 0e-3, # Signal displacement on the x axis [m]
	'curvature_radiusx' : np.array([0e-3,inf]), # Signal's x-radius of curavture center (x,z) in vaccum of the phase front, positive = defoc, negative = foc, inf = collimated [m]
	'Alpha' : 0 # Signal's angle between the k vector and z axis (polar angle) positive angle for positive x [deg]
}

Signal['omega'] = 2*np.pi*c/Signal['Lambda'] #signal central pulsation [rad/s]
Signal['k0'], Signal['k_in_crystal'], Signal['k0_theta'], Signal['k0_phi'] = get_k0_k_ktheta_kphi(Signal['Alpha'], rotation_matrix)
Signal['index'], Signal['EField'] =  refractiveindex(Signal['Lambda'], Signal['k0_theta'][0],Signal['k0_theta'][0],param_crystal['crystal'],Type[1])
Signal['k0'] = 2*np.pi/Signal['Lambda']*Signal['k0']*Signal['index'] # Signal's k vector in laboratory frame
Signal['index'] = Signal['index']*np.ones((1,Nx)) # initializing refractive index vector of Signal

pulse_signal = create_beam2Dxt(Signal, X, Omega)


Idler = {
	'Energy' : 0, #Energy per meter of the Idler [J/m^2]
	'Lambda' : (1/Pump['Lambda'] - 1/Signal['Lambda'])**(-1), #Idler's central wavelength [m]
	'FWHMt' : 40e-15, # Idler's FWHM in Intensity in time domain [s]
	'Phi1' : 0e-15, # Idler first order spectral phase (linked to group delay) [s]
	'Phi2' : 0e-30, # Idler second order spectral phase (linked to linear frequency chirp) [s^2]
	'Phi3' : 0e-45, # Idler third order spectral phase [s^3]
	'FWHMx' : 20e-3, # pump's beam diameter on the x axis (Intensity)[m]
	'x0' : 0e-3, # Idler displacement on the x axis [m]
	'curvature_radiusx' : np.array([0e-3,inf]), # Idler's x-radius of curavture center (x,z) in vaccum of the phase front, positive = defoc, negative = foc, inf = collimated [m]
	'Alpha' : 0, # Idler's angle between the k vector and z axis (polar angle) positive angle for positive x [deg]
}

Idler['omega'] = 2*np.pi*c/Idler['Lambda'] #Idler central pulsation [rad/s]
Idler['k0'], Idler['k_in_crystal'], Idler['k0_theta'], Idler['k0_phi'] = get_k0_k_ktheta_kphi(Idler['Alpha'], rotation_matrix)
Idler['index'], Idler['EField'] =  refractiveindex(Idler['Lambda'], Idler['k0_theta'][0],Idler['k0_theta'][0], param_crystal['crystal'],Type[2])
Idler['k0'] = 2*np.pi/Idler['Lambda']*Idler['k0']*Idler['index'] # Idler's k vector in laboratory frame
Idler['index'] = Idler['index']*np.ones((1,Nx)) # initializing refractive index vector of Idler

pulse_idler = create_beam2Dxt(Idler, X, Omega)

EField_combi = np.array([\
					Pump['EField'][0,0]*Signal['EField'][0],
					Pump['EField'][1]*Signal['EField'][1],
					Pump['EField'][2]*Signal['EField'][2],
					Pump['EField'][1]*Signal['EField'][2] + Pump['EField'][2]*Signal['EField'][1],
					Pump['EField'][0]*Signal['EField'][2] + Pump['EField'][2]*Signal['EField'][0],
					Pump['EField'][0]*Signal['EField'][1] + Pump['EField'][1]*Signal['EField'][0] 
    ]) 

# vector needed to multiply with nonlinear matrix (see p41 Boyd)

P = d.dot(EField_combi) #polarisation of idler
P = np.transpose(P).dot(Idler['EField']/MyNorm((Idler['EField']))) #projection of polarisation on Electric field of idler
deff = MyNorm(P)/MyNorm((Pump['EField']))/MyNorm((Pump['EField'])) #scalar value of the nonlinear matrix [m/V]

deltakz = Pump['k0'][2,0] - Signal['k0'][2,0] - Idler['k0'][2,0]

k1_pump, k2_pump = get_k1_k2(Pump, param_crystal, Type[0], rotation_matrix, kx_vect, Nx, derivative_step_omega, max_error_ref_index)
k1_signal, k2_signal = get_k1_k2(Signal, param_crystal, Type[1], rotation_matrix, kx_vect, Nx, derivative_step_omega, max_error_ref_index)
k1_idler, k2_idler = get_k1_k2(Idler, param_crystal, Type[2], rotation_matrix, kx_vect, Nx, derivative_step_omega, max_error_ref_index)

## Simulation

Nz = int(np.round(param_crystal['L']/step_z))
z = 0

# amplitude in the kx,ky domain

amplitude_pump = fftshift(fft(ifftshift(pulse_pump,1),Nx,0),0)*resol_x/sqrt(2*np.pi)
amplitude_signal = fftshift(fft(ifftshift(pulse_signal,1),Nx,0),0)*resol_x/sqrt(2*np.pi)
amplitude_idler = fftshift(fft(ifftshift(pulse_idler,1),Nx,0),0)*resol_x/sqrt(2*np.pi)

## NLO PROPAGATION in crystal
# h = waitbar(0,'Simulation of NLO 0%')
# # pourcentage = 0.05

# for i in range(Nz):
# #     # if i/Nz>=pourcentage:
# #         # waitbar(i/Nz,h,['Simulation of NLO ', num2str(round(i/Nz*100)),'%'])
# #         # pourcentage = round(i/Nz*20)/20 + 0.05;
    
    
amplitude_pump,amplitude_signal,amplitude_idler = RK4_NLO_2Dxt(amplitude_pump,amplitude_signal,amplitude_idler,\
        Pump['omega'],Signal['omega'],Idler['omega'],Pump['index'],Signal['index'],Idler['index'],k1_pump,k1_signal,k1_idler,k2_pump,k2_signal,k2_idler,\
        KX,Omega,deff,deltakz,z,step_z)
#     z = z+step_z


# end_pulse_pump = fftshift(fft(ifft(ifftshift(amplitude_pump),Nx,1),Nt,0))*2*np.pi/(resol_x*resol_t*Nt)
# end_pulse_signal = fftshift(fft(ifft(ifftshift(amplitude_signal),Nx,1),Nt,0))*2*np.pi/(resol_x*resol_t*Nt)
# end_pulse_idler = fftshift(fft(ifft(ifftshift(amplitude_idler),Nx,1),Nt,0))*2*np.pi/(resol_x*resol_t*Nt)

# # energy_pump = 0.5*eps0*c*ref_index_pump(floor(Nx/2+1))*trapz(x_vect,trapz(t_vect,abs(end_pulse_pump).^2,1),2)
# # energy_signal = 0.5*eps0*c*ref_index_signal(floor(Nx/2+1))*trapz(x_vect,trapz(t_vect,abs(end_pulse_signal).^2,1),2)
# # energy_idler = 0.5*eps0*c*ref_index_idler(floor(Nx/2+1))*trapz(x_vect,trapz(t_vect,abs(end_pulse_idler).^2,1),2)
# # total_energy = energy_pump + energy_signal + energy_idler

# # plottons les resultats dans le domaine temporel
# fig1 = plt.figure(1)
# plt.clf()
# plt.pcolormesh(x_vect*1e3,t_vect*1e15,abs(end_pulse_pump)**2, cmap='gnuplot2')
# plt.title('pump (800nm)')
# plt.xlabel('position [mm]')
# plt.ylabel('time [fs]')
# plt.colorbar()
# plt.show()

# fig1 = plt.figure(1)
# plt.clf()
# plt.pcolormesh(x_vect*1e3,t_vect*1e15,abs(end_pulse_signal)**2, cmap='gnuplot2')
# plt.title('signal (1800nm)')
# plt.xlabel('position [mm]')
# plt.ylabel('time [fs]')
# plt.colorbar()
# plt.show()

# fig1 = plt.figure(1)
# plt.clf()
# plt.pcolormesh(x_vect*1e3,t_vect*1e15,abs(end_pulse_idler)**2, cmap='gnuplot2')
# plt.title('idler (1300nm)')
# plt.xlabel('position [mm]')
# plt.ylabel('time [fs]')

# plt.colorbar()
# plt.show()