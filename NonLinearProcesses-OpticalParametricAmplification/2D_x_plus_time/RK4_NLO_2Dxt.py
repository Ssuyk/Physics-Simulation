import numpy as np
from numpy.fft import *

def RK4_NLO_2Dxt(A_p0,A_s0,A_i0,omega_p,omega_s,omega_i,n_p,n_s,n_i,k1_p,k1_s,k1_i,k2_p,k2_s,k2_i,KX,Omega,deff,deltak,z0,dz):
	#Runge-Kutta of the 4th order to solve nonlinear optics in the x-t domain
	#   Detailed explanation goes here

	#----Setup for RK4-------
	
	c = 3e8 #speed of light
	dx = -1/2/KX[0,0]*2*np.pi #resolution in x [m] used to scale FFT
	dt = -1/2/Omega[0,0]*2*np.pi #resolution in t [s] used to scale FFT
	Nt = np.shape(KX)[0] 
	Nx = np.shape(KX)[1]


	thetap = np.arccos(np.sqrt((n_p*omega_p/c)**2 - KX[int(np.floor(Nt/2+1)),:]**2)/(n_p*omega_p/c)) #polar angle of the kx,ky grid for the pump
	thetas = np.arccos(np.sqrt((n_s*omega_s/c)**2 - KX[int(np.floor(Nt/2+1)),:]**2)/(n_s*omega_s/c)) #polar angle of the kx,ky grid for the signal
	thetai = np.arccos(np.sqrt((n_i*omega_i/c)**2 - KX[int(np.floor(Nt/2+1)),:]**2)/(n_i*omega_i/c)) #polar angle of the kx,ky grid for the idler

	delkp = omega_p/c*(n_p*(1 - np.sin(thetap)**2/2)-n_p[0,int(np.floor(Nx/2+1))])
	delks = omega_s/c*(n_s*(1 - np.sin(thetas)**2/2)-n_s[0,int(np.floor(Nx/2+1))])
	delki = omega_i/c*(n_i*(1 - np.sin(thetai)**2/2)-n_i[0,int(np.floor(Nx/2+1))])

	a_p0 = A_p0*(np.dot(np.ones((Nt,1)),np.exp(-1j*delkp*z0))) #pump's amplitude without oscillation in z
	# a_s0 = A_s0*(np.ones((Nt,1))*np.exp(-1j*delks*z0)) #signal's amplitude without oscillation in z
	# a_i0 = A_i0*(np.ones((Nt,1))*np.exp(-1j*delki*z0)) #idler's amplitude without oscillation in z

	#-----RK4 NLO XT--------
	# fp = lambda a_p,a_s,a_i,z: 1j*0.5*('np.ones((Nt,1))*k2_p')*Omega**2*a_p + 1j*omega_p*deff/c/(n_p[np.floor(Nx/2+1)])*np.exp(-1j*deltak*z)*(np.ones((Nt,1))*np.exp(-1j*delkp*z))\
	# 		*fftshift(fft(ifft(ifftshift(	fftshift(ifft(fft(ifftshift(a_s*(np.ones((Nt,1))*np.exp(1j*delks*z))),Nt,0),Nx,1))\
	# 		*fftshift(ifft(fft(ifftshift(a_i*(np.ones((Nt,1))*np.exp(1j*delki*z))),Nt,0),Nx,1))	),Nt,0),Nx,1))*2*np.pi/dx/dt/Nt


	# fs = lambda a_p,a_s,a_i,z: 1j*(np.ones((Nt,1))*(k1_s - k1_p[np.floor(Nx/2+1)]))*Omega*a_s + 1j*0.5*(np.ones((Nt,1))*k2_s)*Omega**2*a_s\
	#  		+ 1j*omega_s*deff/c/(n_s[np.floor(Nx/2+1)])*np.exp(1j*deltak*z)*(np.ones((Nt,1))*np.exp(-1j*delks*z))\
	#  		* fftshift(fft(ifft(ifftshift(fftshift(ifft(fft(ifftshift(a_p*(np.ones((Nt,1))*np.exp(1j*delkp*z))),Nt,1),Nx,2))\
	#  		* np.conj(fftshift(ifft(fft(ifftshift(a_i*(np.ones((Nt,1))*np.exp(1j*delki*z))),Nt,1),Nx,2)))),Nt,1),Nx,2))*2*np.pi/dx/dt/Nt


	# fi = lambda a_p,a_s,a_i,z: 1j*(np.ones((Nt,1))*(k1_i - k1_p[np.floor(Nx/2+1)]))*Omega*a_i + 1j*0.5*(np.ones((Nt,1))*k2_i)*Omega**2*a_i\
	#  	+ 1j*omega_i*deff/c/(n_i[np.floor(Nx/2+1)])*np.exp(1j*deltak*z)*(np.ones((Nt,1))*np.exp(-1j*delki*z))\
	#  	* fftshift(fft(ifft(ifftshift(fftshift(ifft(fft(ifftshift(a_p*(np.ones((Nt,1))*np.exp(1j*delkp*z))),Nt,1),Nx,2))\
	# 	* np.conj(fftshift(ifft(fft(ifftshift(a_s*(np.ones((Nt,1))*exp(1j*delks*z))),Nt,1),Nx,2)))),Nt,1),Nx,2))*2*np.pi/dx/dt/Nt


	# k1p = dz*fp(a_p0,a_s0,a_i0,z0)
	# k1s = dz*fs(a_p0,a_s0,a_i0,z0)
	# k1i = dz*fi(a_p0,a_s0,a_i0,z0)

	# k2p = dz*fp(a_p0 + k1p/2,a_s0 + k1s/2,a_i0 + k1i/2,z0 + dz/2)
	# k2s = dz*fs(a_p0 + k1p/2,a_s0 + k1s/2,a_i0 + k1i/2,z0 + dz/2)
	# k2i = dz*fi(a_p0 + k1p/2,a_s0 + k1s/2,a_i0 + k1i/2,z0 + dz/2)

	# k3p = dz*fp(a_p0 + k2p/2,a_s0 + k2s/2,a_i0 + k2i/2,z0 + dz/2)
	# k3s = dz*fs(a_p0 + k2p/2,a_s0 + k2s/2,a_i0 + k2i/2,z0 + dz/2)
	# k3i = dz*fi(a_p0 + k2p/2,a_s0 + k2s/2,a_i0 + k2i/2,z0 + dz/2)

	# k4p = dz*fp(a_p0 + k3p,a_s0 + k3s,a_i0 + k3i,z0 + dz)
	# k4s = dz*fs(a_p0 + k3p,a_s0 + k3s,a_i0 + k3i,z0 + dz)
	# k4i = dz*fi(a_p0 + k3p,a_s0 + k3s,a_i0 + k3i,z0 + dz)

	# a_p1 = a_p0 + 1/6*(k1p + 2*k2p + 2*k3p + k4p)
	# a_s1 = a_s0 + 1/6*(k1s + 2*k2s + 2*k3s + k4s)
	# a_i1 = a_i0 + 1/6*(k1i + 2*k2i + 2*k3i + k4i)

	# test = conj(fftshift(ifft(fft(ifftshift(a_s0*(np.ones((Nt,1))*np.exp(1j*delks*z0))),Nt,1),Nx,2)))*2*np.pi/dx/dt/Nt

	# # figure(1);pcolor(abs(testp).^2);shading interp;cmap =jet_white;colormap(cmap);pause(0.1)
	# # figure(2);pcolor(abs(tests).^2);shading interp;cmap =jet_white;colormap(cmap);pause(0.1)
	# # figure(3);pcolor(abs(testi).^2);shading interp;cmap =jet_white;colormap(cmap);pause()

	# #---Real amplitude----
	# A_p1 = a_p1*(np.ones((Nt,1))*np.exp(1j*delkp*(z0+dz))); # pump's amplitude with oscillation in z
	# A_s1 = a_s1*(np.ones((Nt,1))*np.exp(1j*delks*(z0+dz))); # signal's amplitude with oscillation in z
	# A_i1 = a_i1*(np.ones((Nt,1))*np.exp(1j*delki*(z0+dz))); # idler's amplitude with oscillation in z
	return 0
	# return A_p1, A_s1, A_i1