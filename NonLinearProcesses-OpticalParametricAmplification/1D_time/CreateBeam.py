import numpy as np
from refractiveindex import *

c = 3e8
eps0 = 8.85e-12 #vacuum permittivity [F/m]

def Get_K0_k1_k2(params_beam, ref_index, params_crystal, Type, derivative_step_omega):
	Lambda = np.array([params_beam['Lambda']]) #wavelength [m]
	omega = np.array([params_beam['omega']])
	theta = params_crystal['theta_crystal']
	phi_crystal = params_crystal['phi_crystal']
	crystal = params_crystal['crystal']

	k0 = 2*np.pi/Lambda*ref_index

	ref_index_plus, Efield_plus = refractiveindex(\
		2*np.pi*c/(omega+derivative_step_omega),theta,\
		phi_crystal,crystal,Type)

	ref_index_minus, Efield_minus = refractiveindex(\
		2*np.pi*c/(omega-derivative_step_omega),theta,\
		phi_crystal,crystal,Type)

	k0_plus = (omega+derivative_step_omega)*ref_index_plus/c
	k0_minus = (omega-derivative_step_omega)*ref_index_minus/c

	k1 = (k0_plus - k0_minus)/2/derivative_step_omega
	k2 = (k0_plus - 2*k0 + k0_minus)/derivative_step_omega**2


	return k0, k1, k2


def create_beam(params_beam, params_crystal, Type, nu_vect):
	Fluence = params_beam['Fluence'] #Fluence of the pump [J/m^2]
	Lambda = np.array([params_beam['Lambda']]) #wavelength [m]
	omega = params_beam['omega']
	FWHMt = params_beam['FWHMt']
	Phi1 = params_beam['Phi1']
	Phi2 = params_beam['Phi2']
	Phi3 = params_beam['Phi3']
	theta = params_crystal['theta_crystal']
	phi_crystal = params_crystal['phi_crystal']
	crystal = params_crystal['crystal']

	# principal refractive index of the pump + unitary Efield of pump in crystal frame
	ref_index, Efield =	refractiveindex(Lambda,theta,phi_crystal,crystal,Type)

	# %pump's electric field amplitude A, where E = A*exp(1i*k*z - w*t)
	Amplitude = np.sqrt(2*Fluence*FWHMt*np.sqrt(np.pi)/(c*eps0*ref_index*np.sqrt(np.log(2))))*\
    np.exp(-4*np.pi**2*nu_vect**2*FWHMt**2/8/np.log(2))*np.exp(-1j*(Phi1*2*np.pi*nu_vect + \
    	1/2*Phi2*(2*np.pi*nu_vect)**2 + 1/6*Phi3*(2*np.pi*nu_vect)**3))

	return Amplitude, ref_index, Efield 


# Nt = 2**10 #nnumber of points in time domain
# resol_t = 0.5e-15 #time resolution [s]

# time_vect = resol_t*np.linspace(-Nt/2,Nt/2-1,Nt)
# nu_vect = 1/2/resol_t*np.linspace(-1,1,Nt+1) 
# #pulsation vector not shifted [rad/s]
# nu_vect = nu_vect[:-1]

# param_crystal = {
# 	'crystal' : 'BBO', #string for the nonlinear crystal	
# 	'theta_crystal' : 30.86, #cut angle of the crystal
# 	'phi_crystal' : 0, #cut angle of the crystal
# 	'L' : 2e-3 #Length of crystal
# }
# Type = 'FSF' #type of the interaction {'FSS','FSF','FFS'} (fast=F,slow=S) (i.e type I = FSS)


# param_pump = {
# 	'Fluence': 36, #Fluence of the signal [J/m^2]
# 	'Lambda': 800e-9, #signal's central wavelength [m]
# 	'FWHMt': 40e-15, # signal's FWHM in Intensity in time domain [s]
# 	'Phi1': 0e-15, # signal first order spectral phase (linked to group delay) [s]
# 	'Phi2': 0e-30, # signal second order spectral phase (linked to linear frequency chirp) [s^2]
# 	'Phi3': 0e-45, # signal third order spectral phase [s^3]
# }
# param_pump['omega'] = 2*np.pi*c/param_pump['Lambda'] #signal central pulsation [rad/s]
# ref_index = 1

# derivative_step_omega = 1e12 #step in omega to perform numerical derivative [rad/s]

# k0, k0_plus, k0_minus, k1, k2 = Get_K0_k1_k2(param_pump, ref_index, param_crystal, Type[0], derivative_step_omega)
# amplitude_pump, index_ref_pump, Efield_pump = create_beam(param_pump, param_crystal, Type[0], nu_vect)
    	