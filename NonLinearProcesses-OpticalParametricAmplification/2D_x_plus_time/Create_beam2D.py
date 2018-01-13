import numpy as np
import warnings
from nonlinearcoefficient import *
from refractiveindex import *
import matplotlib.pyplot as plt
c=3e8
eps0 = 8.854e-12

def cosd(Theta):
	return np.cos(np.deg2rad(Theta))

def sind(Theta):
	return np.sin(np.deg2rad(Theta))

def acosd(x):
	return np.rad2deg(np.arccos(x))

def atan2d(A,B):
	return np.rad2deg(np.arctan2(A,B))

def MyNorm(array):
	return np.sqrt(np.sum(array**2))

def get_k0_k_ktheta_kphi(alpha, rotation_matrix):
	k0 = np.array([[sind(alpha)], [0], [cosd(alpha)]]) #unitary pump's k vector in laboratory frame
	k_in_crystal = np.linalg.solve(rotation_matrix, k0) # unitary pump's k vector in crystal frame (missing refractive index)
	k0_theta = acosd(k_in_crystal[2]) # angle between pump k vector and z axis of the crystal frame [deg]
	k0_phi = atan2d(k_in_crystal[1],k_in_crystal[0]) # angle between pump projection of k vector on the xy crystal plane and the x axis [deg]

	return k0, k_in_crystal, k0_theta, k0_phi

def create_beam2Dxt(params_beam, X, Omega):
	Energy = params_beam['Energy'] # Fluence of the pump [J/m]
	Lambda = np.array([params_beam['Lambda']]) #wavelength [m]
	omega = params_beam['omega']
	FWHMt = params_beam['FWHMt']
	Phi1 = params_beam['Phi1']
	Phi2 = params_beam['Phi2']
	Phi3 = params_beam['Phi3']
	FWHMx = params_beam['FWHMx']
	Index = params_beam['index']
	x0 = params_beam['x0']
	CurvX = params_beam['curvature_radiusx'] 
	k0 = params_beam['k0']

	return np.sqrt(2*Energy*FWHMt/(np.pi*Index[0]*eps0*c*FWHMx))\
		*np.exp(-(X-x0)**2*2*np.log(2)/FWHMx**2 - (Omega)**2*FWHMt**2/8/np.log(2))\
    	*np.exp(1j*2*np.pi/Lambda*(X-CurvX[0])**2/CurvX[1])*np.exp(1j*(k0[0]*X))\
    	*np.exp(1j*(Phi1*Omega + 1/2*Phi2*Omega**2 + 1/6*Phi3*Omega**3))

def get_k1_k2(params_beam, params_crystal, Type, rotation_matrix, kx_vect, Nx, derivative_step_omega, max_error_ref_index):

	# the objective is to find the refractive index as function of kx at
	# the central frequency as well as k1 = dk/domega and k2=d^2k/domega^2. The
	# method is an iterative one where the k0 refractive_index is used as a
	# first guess. The iteration stops when the change is less than
	# max_error_ref_index or after 50 steps

	crystal = params_crystal['crystal']  
	omega = params_beam['omega']
	Lambda = np.array(params_beam['Lambda']) #wavelength [m]
	k0 = params_beam['k0']
	index = params_beam['index']

	k1 = np.zeros((1,Nx)) #pump's first derivative of k vector with respect of omega
	k2 = np.zeros((1,Nx)) #pump's second derivative of k vector with respect of omega

	for i in range(Nx):
		# ----pump----
		# first guess
		if i==0:
			temp_k_norm = MyNorm(k0) # norm of the pump's principal k direction
		else:
			temp_k_norm = 2*np.pi/Lambda*index[0,i-1] #previous most relevant answer

		temp_kk = np.array([[kx_vect[i]], [0], [np.sqrt(temp_k_norm**2-kx_vect[i]**2)]]) #unitary pump's k vector in laboratory frame
		temp_k = np.linalg.solve(rotation_matrix, temp_kk/temp_k_norm)# unitary k vector in crystal frame
		temp_theta = acosd(temp_k[2]) # polar angle of the k vector in the crystal frame
		temp_phi = atan2d(temp_k[1],temp_k[0]) # azimuthal angle of the k vector in the crystal frame
		temp_ref_index, Etrash  = refractiveindex(Lambda,temp_theta[0],temp_phi[0],crystal,Type) # refractive index in the direction of the k vector

		n = 1 #counter
		# iteration on refractive_index as a fct of kx and ky
		while(temp_ref_index - index[0,i]) > max_error_ref_index or n>50 :
			index[0,i] = temp_ref_index
			temp_k_norm = 2*np.pi*index[0,i]/Lambda #norm of k vector
			temp_k = np.linalg.solve(rotation_matrix, np.array([[kx_vect[i]], [0], [np.sqrt(temp_k_norm**2-kx_vect[i]**2)]])/temp_k_norm) # unitary k vector in crystal frame
			temp_theta = acosd(temp_k[2]) # angle between k vector and crystal z axis
			temp_phi = atan2d(temp_k[1],temp_k[0]) #angle between the projection of the k vector on the xy plane and x axis of the crystal
			temp_ref_index, Etrash = refractiveindex(Lambda,temp_theta[0],temp_phi[0],crystal,Type) # refractive index in the direction of the k vector
			n = n + 1 #counter

		index[0,i] = temp_ref_index
		temp_k_norm = 2*np.pi*index[0,i]/Lambda #norm of k vector

		if n>50:
			warnings.warn('Iteration could not converge for the refractive index of the pump')

		ref_index_plus, Etrash = refractiveindex(2*np.pi*c/(omega+derivative_step_omega),temp_theta[0],temp_phi[0],crystal,Type)
		ref_index_minus, Etrash = refractiveindex(2*np.pi*c/(omega-derivative_step_omega),temp_theta[0],temp_phi[0],crystal,Type)
		k0_plus = (omega+derivative_step_omega)*ref_index_plus/c
		k0_minus = (omega-derivative_step_omega)*ref_index_minus/c
		k1[0,i] = (k0_plus - k0_minus)/2/derivative_step_omega #first derivative of pump's k vector along omega
		k2[0,i] = (k0_plus - 2*temp_k_norm + k0_minus)/derivative_step_omega**2 #second derivative of pump's k vector along omega

	return k1, k2 
    

## Test 

# def RotMat(fast_axis_orientation, theta_crystal, phi_crystal):
# 	return np.dot(np.array([[cosd(fast_axis_orientation), sind(fast_axis_orientation), 0], [-sind(fast_axis_orientation), cosd(fast_axis_orientation), 0], [0, 0, 1]])\
# 	, np.dot(np.array([ [cosd(theta_crystal), 0, -sind(theta_crystal)], [0, 1, 0], [sind(theta_crystal), 0, cosd(theta_crystal)] ]),\
# 	np.array([ [cosd(phi_crystal), sind(phi_crystal), 0], [-sind(phi_crystal), cosd(phi_crystal), 0], [0, 0, 1] ])))    


# Nx = 2**8 #number of points in x
# Nt = 2**10 #number of points in time domain
# resol_x = 750e-6 #space resolution [m]
# resol_t = 1e-15 #time resolution [s]

# x_vect = resol_x*np.linspace(-Nx/2,Nx/2-1,Nx)
# time_vect = resol_t*np.linspace(-Nt/2,Nt/2-1,Nt)

# X,T = np.meshgrid(x_vect, time_vect)

# kx_vect = 2*np.pi/2/resol_x*np.linspace(-1,1, Nx+1) #pulsation vector +1 points in x [rad/m]
# kx_vect = kx_vect[:-1]

# omega_vect = 2*np.pi/2/resol_t*np.linspace(-1,1,Nt+1) 
# omega_vect = omega_vect[:-1]

# [KX, Omega] = np.meshgrid(kx_vect, omega_vect)
# [X2, Omega2] = np.meshgrid(x_vect, omega_vect)

# max_error_ref_index = 1e-5; #maximum error tolerated on the value of the refractive index
# derivative_step_omega = 1e12 #step in omega to perform numerical derivative [rad/s]

# Pump = {
# 	'Energy' : 2.26, #Energy per meter of the signal [J/m^2]
# 	'Lambda' :  np.array([800e-9]), #signal's central wavelength [m]
# 	'FWHMt' : 40e-15, # signal's FWHM in Intensity in time domain [s]
# 	'Phi1' : 0e-15, # signal first order spectral phase (linked to group delay) [s]
# 	'Phi2' : 0e-30, # signal second order spectral phase (linked to linear frequency chirp) [s^2]
# 	'Phi3' : 0e-45, # signal third order spectral phase [s^3]
# 	'FHWMx' : 20e-3, # pump's beam diameter on the x axis (Intensity)[m]
# 	'x0' : 0e-3, # pump displacement on the x axis [m]
# 	'curvature_radiusx' : np.array([0e-3, 1e20]), # pump's x-radius of curvature center (x,z) in vaccum of the phase front, positive = defoc, negative = foc, inf = collimated [m]
# 	'Alpha' : 0 # pump's angle between the k vector and z axis (polar angle) positive angle for positive x [deg]
# }
# #--------Crystal info--------
# param_crystal = {
# 	'crystal' : 'BBO', #string for the nonlinear crystal	
# 	'theta_crystal' : 30.86, #cut angle of the crystal
# 	'phi_crystal' : 0, #cut angle of the crystal
# 	'L' : 1.5e-3, #Length of crystal
# 	'fast_axis_orientation' : 0 #fast axis orientation compare to x axis laboratory frame
# }
# Type = 'FSS' #type of the interaction {'FSS','FSF','FFS'} (fast=F,slow=S) (i.e type I = FSS)

# d = nonlinearCoefficientBBO()*1e-12 #non linear coefficient matrix [m/V]
# rotation_matrix = RotMat(param_crystal['fast_axis_orientation'], param_crystal['theta_crystal'], param_crystal['phi_crystal'])

# Pump['omega'] = 2*np.pi*c/Pump['Lambda'] #signal central pulsation [rad/s]
# Pump['k0'], Pump['k_in_crystal'], Pump['k0_theta'], Pump['k0_phi'] = get_k0_k_ktheta_kphi(Pump['Alpha'], rotation_matrix)
# Pump['index'], Pump['EField'] =  refractiveindex(Pump['Lambda'], Pump['k0_theta'][0],Pump['k0_theta'][0],param_crystal['crystal'],Type[0])
# Pump['k0'] = 2*np.pi/Pump['Lambda']*Pump['k0']*Pump['index'] # pump's k vector in laboratory frame
# Pump['index'] = Pump['index']*np.ones((1,Nx)) # initializing refractive index vector of pump

# pulse_pump = create_beam2Dxt(Pump, X, Omega)

# k1_pump, k2_pump = get_k1_k2(Pump, param_crystal, Type[0], rotation_matrix, kx_vect, Nx, derivative_step_omega, max_error_ref_index)
# fig1 = plt.figure(1)
# plt.pcolormesh(x_vect,time_vect,np.abs(pulse_pump)**2)
# plt.show()
