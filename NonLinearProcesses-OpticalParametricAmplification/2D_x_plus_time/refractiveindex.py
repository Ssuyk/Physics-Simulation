import numpy as np
from sellmeierBBO import *

def cosd(Theta):
	return np.cos(Theta*np.pi/180)

def sind(Theta):
	return np.sin(Theta*np.pi/180)

def MyNorm(array):
	return np.sqrt(np.sum(array**2))

def refractiveindex(Lambda, Theta, Phi, crystal, Type):

	"""
	Calculate the refractive index for a given wavelength and crystal
	orientation. 
	Input :
	- Lambda : wavelength in m 
	- Theta angle between the cristal z axis and the k vector (deg)
	- phi: angle between the cristal x axis and the projection 
	  of k vector on the XY plane (deg)
	- crystal: string for the cristal. Ex: 'KTA'
	- type: either 'F' for fast or 'S' for slow

	Output:
	-n : refractive index
	-E : unitary Electric field orientation in crystal frame

	"""
	n = np.zeros((1,np.size(Lambda)))
	E = np.zeros((3,np.size(Lambda)))
	nx = sellmeier(Lambda, crystal, 'x')
	ny = sellmeier(Lambda, crystal, 'y')
	nz = sellmeier(Lambda, crystal, 'z')
	kvect = np.array([[sind(Theta)*cosd(Phi), sind(Theta)*sind(Phi),cosd(Theta)]])

	for i in range(np.size(Lambda)):
		eta = np.array([[1/nx[i]**2, 0, 0],\
					[0,1/ny[i]**2,0],\
					[0,0, 1/nz[i]**2]])
		# print(eta)
		#based on Boyd derivation/page 43.
		n_vect, D_mat = np.linalg.eig((np.eye(3)-\
			(kvect.conj().T).dot(kvect)).dot(eta))
		
		
		order = np.argsort(-n_vect)
		n_vect = n_vect[order]
		# minus sign in argsort for sorting with descending order
		
		#problem avec order et dmat, a corriger!!!!
		D_mat = D_mat[:,order]
		
		if Type.upper() == 'F':
			n[0,i] = np.sqrt(1/n_vect[0])
			E[:,i] = eta.dot(D_mat[:,0])
			E[:,i] = E[:,i]/MyNorm(E[:,i]) 
			if np.isinf(abs(n[0][i])) or np.isnan(abs(n[0][i])):
				raise ValueError('invalid wavelength to calculate the refractive index')
		elif Type.upper() == 'S':
			n[0,i] = np.sqrt(1/n_vect[1])
			E[:,i] = eta.dot(D_mat[:,1])
			E[:,i] = E[:,i]/MyNorm(E[:,i])
			if np.isinf(abs(n[0][i])) or  np.isnan(abs(n[0][i])):
				raise ValueError('invalid wavelength to calculate the refractive index')
		else:
			raise ValueError('Type must be F or S')

	return n, E				


# test
# Lambda = np.array([1440e-9])
# Theta = 30.86
# Phi = 0
# crystal = 'BBO'
# Type = 'S'
# index, Field,D_mat = refractiveindex(Lambda, Theta, Phi, crystal, Type)