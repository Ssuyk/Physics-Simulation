# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 21:56:58 2017

@author: Vincent
"""

import numpy as np
from matplotlib import pyplot as pl
from numpy import fft as ft

def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def gaussian_beam_cart(X, Y, A, B, x0, y0, Sigma_X, Sigma_Y):
    return A+B*np.exp(-((X-x0)/(2*Sigma_X))**2+-((Y-y0)/(2*Sigma_Y))**2)
    
def gaussian_beam_pol(R, A, B, r0, SigmaR):
    return A+B*np.exp(-((R-r0)/(2*SigmaR))**2)
    
N_pts = 1024    
x = np.linspace(-10,10, N_pts)
y = x
delta_x = x[1]-x[0]
r = np.sqrt(x**2+y**2)
[X, Y] = np.meshgrid(x,y)
[R,Theta] = cart2pol(X,Y)

Lambda= 800*10**(-9) #m
k0 = 2*np.pi/Lambda
zL= 1 #propagation distance from mask to lens
F = 1 #Focal length
zP = 1 #Propagation Distance after lens


InitBeam = gaussian_beam_pol(R, 0, 1, 0, 0.01)

fig1 = pl.figure(1,**(dict(figsize=(6,5), dpi=64) ))
pl.imshow( np.abs(InitBeam)**2,aspect='auto', interpolation='none', extent=extents(x) + extents(y), origin='lower')
pl.xlabel('Period', fontsize = 14)
pl.title('Init Beam', fontsize = 14)
##------------------------------------------------------------------------
## \\\\\\\\\\\\\\\\\\ Propagation from the mask to the lens\\\\\\\\
##------------------------------------------------------------------------
r_out = ft.fftshift(ft.fftfreq(N_pts, x[1]-x[0]))*2.*np.pi #Scaling of the output grid obtained through the Fourier Transform  
r_out=r_out-np.mean(r_out);
[uLensArray, vLensArray]=np.meshgrid(r_out,r_out); #output grid

UTF0=ft.fftshift(ft.fft2(ft.fftshift(InitBeam)))*np.exp(-1j*np.pi*Lambda*(uLensArray**2+vLensArray**2)*zL);
ULens=ft.fftshift(ft.ifft2(ft.fftshift(UTF0)));

fig1 = pl.figure(2,**(dict(figsize=(6,5), dpi=64) ))
pl.imshow( np.abs(ULens)**2,aspect='auto', interpolation='none', extent=extents(x) + extents(y), origin='lower')
pl.xlabel('Period', fontsize = 14)
pl.title('Before lens', fontsize = 14)
pl.xlim(-0.1, 0.1)
pl.ylim(-0.1, 0.1)

##------------------------------------------------------------------------
## \\\\\\\\\\\\\\\\\\ Propagation from the lens to the z position \\\\\\\\
##------------------------------------------------------------------------
#Computation of the field at abcissa z after the lens
# It should be chosen close to focus for the FT method to work
#Usual scaling factors
Scale=delta_x/np.sqrt(2*np.pi)    #Amplitude scaling factor for 1D
Scale=Scale**2  # Here is for 2D. Checked on a gaussian:correct
#Here it is the FT taken at x'*k/zP=x'*2*pi/lambda/z. So the linscale and scale requires
#rescaling:
r_out = ft.fftshift(ft.fftfreq(N_pts, x[1]-x[0]))*2.*np.pi #Scaling of the output grid obtained through the Fourier Transform  
r_out =r_out *Lambda/2/np.pi*zP
r_out =r_out -np.mean(r_out)
#Just 2pi is missing on the Scale to fit with the definition of the FFT:
Scale=Scale*(2*np.pi)

[XOutArray, YOutArray]=np.meshgrid(r_out ,r_out)#output grid
ROutArray=np.sqrt(XOutArray**2+YOutArray**2)
U0temp = ULens*np.exp(-1j*k0*(R)**2/2*(1/F-1/zP)) # Input field times a lens and the other one in the Fresnel integral
U0Out=Scale/(Lambda*zP)*ft.fftshift(ft.fft2(ft.fftshift(U0temp))*np.exp(1j*k0*(ROutArray)**2/2/zP));

fig3 = pl.figure(3,**(dict(figsize=(6,5), dpi=64) ))
pl.imshow( np.abs(U0Out)**2,aspect='auto', interpolation='none', extent=extents(r_out) + extents(r_out), origin='lower')
#pl.xlabel('Period', fontsize = 14)
pl.xlim(-0.0001, 0.0001)
pl.ylim(-0.0001, 0.0001)
pl.title('After lens', fontsize = 14)

