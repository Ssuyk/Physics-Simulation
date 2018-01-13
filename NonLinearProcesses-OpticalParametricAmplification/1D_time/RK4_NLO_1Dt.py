import numpy as np

def RK4_NLO_1Dt(A_p0,A_s0,A_i0,omega_p,omega_s,omega_i,n_p,n_s,n_i,k1_p,k1_s,k1_i,k2_p,k2_s,k2_i,omega_vect,deff,deltak,z0,dz):
	'''
	Input:
	- A_p0, A_s0,A_i0 : beam amplitude of the pump, signal and idler in the spectral domain. 
	- omega_p,_s,_i : central frequencies
	- n_p, n_s, n_i : refractive index
	- k1_p, k1_s, k1_i : dk/dw ==> inverse de la vitesse de groupe (1/v_g)
	- k2_p, k2_s, k2_i : d2k/dw2 ==> GVD
	- omega_vect : vecteur frequence
	- deff : nonlinear polarization
	- deltak : phase matching
	- z0 : current z position
	- dz : step for calculations
	
	Output: 
	- A_p1, A_s1, A_i1 : amplitude in spectral domain after propagation through width dz.

	What's going on: 
	- cette fonction est un solver ode couplé pour la propagation non linéaire, en prenant en compte la durée d'impulsion.	
	- les trois équations couplées décrivent la propagation du faisceau pompe (subscript p), signal (subscript s) et idler (subscript i)
	- les trois fonctions lambda fp; fs et fi representent les equations differentielles.
	- Une méthode de Runge-Kutta d'ordre 4 est utilisé (Wikipedia est ton ami).
	- Le solver se fait dans le domaine z, omega. 

	'''


	c = 3e8
	dt = -1/2/omega_vect[0]*2*np.pi

	Nt = np.size(omega_vect)

	fp = lambda A_p,A_s,A_i,z: 1j*0.5*k2_p*omega_vect**2*A_p + 1j*omega_p*deff/c/n_p*np.exp(-1j*deltak*z)\
	    *np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(	np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A_s)))\
	    	*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A_i)))	)))*np.sqrt(2*np.pi)/dt/Nt

	fs = lambda A_p,A_s,A_i,z: 1j*(k1_s - k1_p)*omega_vect*A_s + 1j*0.5*k2_s*omega_vect**2*A_s + 1j*omega_s*deff/c/n_s*np.exp(1j*deltak*z)\
	    *np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(	np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A_p)))\
	    	*np.conj(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A_i))))	)))*np.sqrt(2*np.pi)/dt/Nt     

	# fi = lambda A_p,A_s,A_i,z: 1j*(k1_i/omega_s - k1_p/omega_p)*omega_vect*A_i + 1j*0.5*k2_i*omega_vect**2*A_i + 1j*omega_i*deff/c/n_i*np.exp(1j*deltak*z)\
	fi = lambda A_p,A_s,A_i,z: 1j*(k1_i - k1_p)*omega_vect*A_i + 1j*0.5*k2_i*omega_vect**2*A_i + 1j*omega_i*deff/c/n_i*np.exp(1j*deltak*z)\
	    *np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(	np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A_p)))\
	    	*np.conj(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A_s))))	)))*np.sqrt(2*np.pi)/dt/Nt   

	k1p = dz*fp(A_p0,A_s0,A_i0,z0)
	k1s = dz*fs(A_p0,A_s0,A_i0,z0)
	k1i = dz*fi(A_p0,A_s0,A_i0,z0)

	k2p = dz*fp(A_p0 + k1p/2, A_s0 + k1s/2, A_i0 + k1i/2, z0 + dz/2)
	k2s = dz*fs(A_p0 + k1p/2, A_s0 + k1s/2, A_i0 + k1i/2, z0 + dz/2)
	k2i = dz*fi(A_p0 + k1p/2, A_s0 + k1s/2, A_i0 + k1i/2, z0 + dz/2)

	k3p = dz*fp(A_p0 + k2p/2, A_s0 + k2s/2, A_i0 + k2i/2, z0 + dz/2)
	k3s = dz*fs(A_p0 + k2p/2, A_s0 + k2s/2, A_i0 + k2i/2, z0 + dz/2)
	k3i = dz*fi(A_p0 + k2p/2, A_s0 + k2s/2, A_i0 + k2i/2, z0 + dz/2)

	k4p = dz*fp(A_p0 + k3p, A_s0 + k3s, A_i0 + k3i, z0 + dz)
	k4s = dz*fs(A_p0 + k3p, A_s0 + k3s, A_i0 + k3i, z0 + dz)
	k4i = dz*fi(A_p0 + k3p, A_s0 + k3s, A_i0 + k3i, z0 + dz)

	A_p1 = A_p0 + 1/6*(k1p + 2*k2p + 2*k3p + k4p)
	A_s1 = A_s0 + 1/6*(k1s + 2*k2s + 2*k3s + k4s)
	A_i1 = A_i0 + 1/6*(k1i + 2*k2i + 2*k3i + k4i)

	return A_p1, A_s1, A_i1
