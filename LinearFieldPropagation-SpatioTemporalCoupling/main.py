# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:54:42 2017

@author: Boulot
"""

import os
path = r'C:\Users\Boulot\Google Drive\Python\INRS\Propagation Faisceau'
os.chdir(path)

import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Boulot\Google Drive\Python\INRS\Propagation Faisceau\Sellmeier_PPLN')


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap

import refractive_index as RI

from CalculateQout import *
from ChangeDomain import *
from Q_matrix import *
from basic_Kmatrix import *



Npts=2**10+1
c=2.997*10**8#m/s
xmax=0.010#meter
tmax=3000#fs
x=np.linspace(-xmax,xmax,Npts)
t=np.linspace(-tmax,tmax,Npts)

Lambda=1.9*10**(-6)#m
f0=c/Lambda*1e-15
omega0=2*np.pi*f0

dt = np.mean(np.diff(t))
#Fs = 1/dt
#df = Fs/Npts
freq = (2*np.pi/(dt)/(Npts-1))*np.linspace(-Npts/2,Npts/2-1,Npts)+f0
#freq = -Fs/2+df:df:Fs/2 + f0;
Lambda_vec = (c/(freq))
Omega=2*np.pi*freq

delta_x=2*xmax/Npts
k=(2*np.pi/(delta_x)/(Npts-1))*np.linspace(-Npts/2,Npts/2-1,Npts)

[X,T]=np.meshgrid(x,t)
[X2,OMEGA]=np.meshgrid(x,Omega)
[K,OMEGA2]=np.meshgrid(k,Omega)
[K2,T2]=np.meshgrid(k,t)

Delta_phi = RI.Sellmeier_LITA_M_freq(OMEGA, 25)

#definition of the initial pulse

E0=1 #to be defined

#%spatially 
R=1e5#m
w=0.5e-2#m

#%temporally

tFWHM=20;
Tau=tFWHM/np.sqrt(np.log(2))/2# pulse width in fs (the formalism use the half width at 1/e2,
#% Enter the FWHM value and it will convert it to 1/e2)
Beta=0.000#temporal chirp

DeltaOmegaFWHM=1/Tau*2*np.sqrt(np.log(2))
DeltaOmegaFWat1_over_e2=4/Tau
DeltaLambdaFWHM=(DeltaOmegaFWHM/omega0*Lambda)*1e9#nm
DeltaLambdaFWat1_over_e2=(DeltaOmegaFWat1_over_e2/omega0*Lambda)*1e9#nm

#No pulse front tilt initially
PFT=0

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
zR=np.pi*w**2/Lambda
waistatfocus=Lambda*f/np.pi/w/np.sqrt(1+f**2/zR**2);

deltaL1=0;#m
deltaL2=0.0;#m
#propagation
L1=f+deltaL1;#m
L2=f+deltaL2;#m

#Define Q/Field/Qout
[Qin,Qminus1]=defineQmatrix(Lambda,R,w,Tau,Beta,PFT)
Rin=XT_2_XOmega(Qminus1,Lambda)
#S_in=XT_2_KOmega(Qminus1, Lambda);
#P_in=XT_2_KT(Qminus1, Lambda);
#
E_TD = E0*np.exp(-1j*np.pi/Lambda*(Qminus1[0,0]*X**2 + 2*Qminus1[0,1]*X*T-Qminus1[1,1]*T**2))*np.exp(1j*omega0*T)
           
E_XOmega=np.exp(-1j*np.pi/Lambda*(Rin[0,0]*X2**2+2*Rin[0,1]*X2*OMEGA-Rin[1,1]*OMEGA**2));
#E_KOmega=exp(-1i*pi/Lambda*(S_in(1,1)*K.^2+2*S_in(1,2).*K.*(OMEGA2)-S_in(2,2)*(OMEGA2).^2));
#E_KT=exp(-1i*pi/Lambda*(P_in(1,1)*K2.^2+2*P_in(1,2).*K2.*T2-P_in(2,2)*T2.^2));

#We go up to the FP 
Temp_Matrix=Freespace(L1,0)*lens(f)* \
     Freespace(L1,0)*grating(Theta_i, Theta_d, Lambda)

#Total_Matrix=grating(Theta_d+deltaTheta_dout, Theta_iout, Lambda)*Freespace(L2,0)* \
#    lens(f)*Freespace(L1+L2,0)*lens(f)* \
#     Freespace(L1,0)*grating(Theta_i, Theta_d, Lambda)
 
Q_temp=CalculateQoutfrom_K_and_Qin(Qin,Temp_Matrix,Lambda)
Q_tempminus1=inv(Q_temp)
R_temp=XT_2_XOmega(Q_tempminus1,Lambda)
S_temp=XT_2_KOmega(Q_tempminus1, Lambda)
P_temp=XT_2_KT(Q_tempminus1, Lambda)
 
E_TD_Temp=E0*np.exp(-1j*np.pi/(Lambda/2)*(Qoutminus1[0,0]*X**2+2*Qoutminus1[0,1]*X*T-Qoutminus1[1,1]*T**2))*np.exp(1j*omega0*T)
E_XOmega_FFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_TD_Temp),n = Npts, axis=1))
E_XOmega_FFT *= np.exp(1j*Delta_phi)
E_XOmega_FFT /= np.max(E_XOmega_FFT)
E_TD_temp_FFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_XOmega_FFT),n = Npts, axis=1))
E_TD_temp_FFT /=np.max(E_TD_temp_FFT)
 
E_XOmega_Temp=np.exp((Rout[0,0]*X2**2+2*Rout[0,1]*X2*(OMEGA-omega0)-Rout[1,1]*(OMEGA-omega0)**2))*np.exp(1j*Delta_phi)
#E_KOmega_Temp=np.exp((Sout[0,0]*K**2 + 2*Sout[0,1]*K*(OMEGA2-omega0) - Sout[1,1]*(OMEGA2-omega0)**2))
#% E_KT_out=exp((Pout(1,1)*K2.^2+2*Pout(1,2).*K2.*T2-Pout(2,2)*T2.^2));
#
Total_Matrix=grating(Theta_d+deltaTheta_dout, Theta_iout, Lambda)*Freespace(L2,0)* \
             lens(f)*Freespace(L2,0)
#%% OUTPUT

fig = plt.figure(1)
plt.clf()

plt.subplot(1,2,1)
plt.pcolormesh(X*100, T, np.abs(E_TD_temp_FFT)**2, rasterized=True, cmap='gnuplot2', vmin=0, vmax=1)
#plt.axis([-0.8, 0.8 , 0.8, 1.2])
#plt.colorbar()
plt.title('(x,w)')
plt.ylabel('Pulsation (rad/fs)')
plt.xlabel('size (cm)')

plt.subplot(1,2,2)
plt.pcolormesh(X*100, T, np.abs(E_TD_Temp)**2, rasterized=True, cmap='gnuplot2', vmin=0, vmax=1)

#plt.pcolormesh(X*100, T, np.abs(E_TD_Final)**2, cmap='gnuplot2', vmin=0, vmax=1)
#plt.axis([-0.5, 0.5, -2000, 2000])
#% imagesc(x*1e2,t,(angle(E_TD_Final)))
#imagesc(x*1e2,t,abs(E_TD_Final).^2)
plt.xlabel('size (cm)')
plt.ylabel('Pulse duration (fs)')
plt.title('(x,t)')

#set(gca,'box','on','linewidth',1.3,'fontsize',14,'fontname','gill sans mt')
#% 
#% subplot(222)
#% % imagesc(x*1e2,Omega,(angle(E_XOmega_out)))
#% imagesc(x*1e2,Omega,abs(E_XOmega_out).^2)
#% xlabel('Beam size (cm)')
#% ylabel('Pulsation (rad.fs-1)')
#% axis tight
#% axis square
#% grid on
#% title('(x,\omega)')
#% set(gca,'box','on','linewidth',1.3,'fontsize',14,'fontname','gill sans mt')
#% ylim([-0.3 0.3])
#% xlim([-2 2])
#% 
#% subplot(223)
#% % imagesc(k*1e-5,t,(angle(E_KT_out)))
#% imagesc(k*1e-5,t,abs(E_KT_out).^2)
#% xlabel('k (10^{5} rad.m^{-1}) ')
#% ylabel('Pulse duration (fs)')
#% axis tight
#% axis square
#% grid on
#% title('(k,t)')
#% set(gca,'box','on','linewidth',1.3,'fontsize',14,'fontname','gill sans mt')
#% % xlim([-0.05 0.05])
#% % ylim([-100 100])
#% 
#% subplot(224)
#% % imagesc(k*1e-5,Omega,(angle(E_KOmega_out)))
#% imagesc(k*1e-5,Omega,abs(E_KOmega_out).^2)
#% xlabel('k (10^{5} rad.m^{-1}) ')
#% ylabel('Pulsation (rad.fs-1)')
#% axis tight
#% axis square
#% grid on
#% title('(k,\omega)')
#% set(gca,'box','on','linewidth',1.3,'fontsize',14,'fontname','gill sans mt')
#% ylim([-0.1 0.1])
#% xlim([-0.05 0.05])
#%% test fft

Npts=1024*4
tmax=100
t=np.linspace(-tmax,tmax,Npts)
delta_t=np.mean(np.diff(t))
Omega=(2*np.pi/(delta_t)/(Npts-1))*np.linspace(-Npts/2,Npts/2-1,Npts)

#tFWHM=18
#Tau2=tFWHM/np.sqrt(np.log(2))/2
#Omega2=1/Tau2

#func=np.exp(-(t/(Tau2*np.sqrt(2)))**2)

My_cosinus=np.cos(4*t)

#func2=np.exp(-(Omega/Omega2/np.sqrt(2))**2)
fftfunc=np.fft.fftshift(np.fft.fft(np.fft.fftshift(My_cosinus),n =Npts, axis = -1))
#fftm1=np.fft.fftshift(np.fft.fft(np.fft.fftshift(func2)))
#
#cfun_Func=fit(Omega',func2','gauss1')
#
plt.figure(2)
plt.clf()
plt.subplot(211)
plt.plot(t, My_cosinus)
#plt.plot(t, (np.abs(fftm1)**2)/np.max((np.abs(fftm1)**2)))
#plt.axis([-30, 30, 0, 1])
plt.grid()

plt.subplot(212)
plt.plot(Omega,abs(fftfunc)**2/max(abs(fftfunc)**2))
#plt.plot(Omega,func2**2)
plt.grid()
#%% test int�gration
E_T=np.trapz(np.abs(E_TD)**2,t,axis = 1)
E_X=np.trapz(np.abs(E_TD)**2,x,axis = 0)
E_Omega=np.trapz(np.abs(E_XOmega)**2, axis = 1)
#% E_X=trapz(abs(E_XOmega).^2,1);
#
E_Omega_out=np.trapz(np.abs(E_XOmega_out)**2,Omega, axis = 1)
E_T_out=np.trapz(np.abs(E_TD_Final)**2,t, axis = 1)
#
plt.figure(3)
plt.plot(E_Omega_out)
#% plot([-0.3 0.3], [0.5 0.50],'linewidth',1.5)
#% plot([-5 5], [exp(-2) exp(-2)],'linewidth',1.5)
#grid on
#% xlim([-200 200])
#% xlabel('Beam size (cm)')
#% ylabel('Intensity (arb. units)')
#%%
#subplot(212)
#plot(x*100,normalise(E_X_out),'linewidth',1.5);hold on
#plot([-5 5], [0.5 0.50],'linewidth',1.5)
#plot([-5 5], [exp(-2) exp(-2)],'linewidth',1.5)
#grid on
#xlim([-3 3])
#% plot([-100 100], [exp(-2) exp(-2)])
#% plot([-100 100], [0.5 0.5])
#% plot([-30 -30], [0 1])
#% plot([30 30], [0 1])
#%%
#% plot(Omega,normalise(E_XT_intX),'linewidth',1.5);
#xlim([-100 100])
#grid on
#subplot(212)
#plot(Lambda_vec,normalise(E_Omega),'linewidth',1.5); hold on
#plot([2600 3400], [0.5 0.5])
#plot([2600 3400], [exp(-2) exp(-2)])
#
#xlabel('Omega')
#ylabel('Signal (unit� arb.)')
#grid on
#xlim([2600 3400])
#
#%% Wigner and Spectrogram
#E_T_for_wigner=trapz(E_TD,2);
#E_T_out_for_wigner=trapz(E_TD_Final,2);
#Wigner_in = mywigner(E_T_for_wigner);
#Wigner_out = mywigner(E_T_out_for_wigner);
#%%
#
#figure_docked('Wigner Function');clf; hold on
#imagesc(t,Omega,normalise(Wigner_out))
#colorbar
#xlim([-500 500])
#ylim([-0.2 0.2])
#%%
#Int_Wigner_TD=trapz(Wigner_in,2);
#Int_Wigner_WD=trapz(Wigner_in,1);
#Int_Wigner_TD_out=trapz(Wigner_out,2);
#Int_Wigner_WD_out=trapz(Wigner_out,1);
#%%
#figure_docked('test_int_wigner');clf;hold on
#subplot(221)
#plot(t,normalise(Int_Wigner_TD))
#xlim([-100 100])
#subplot(222)
#plot(Omega,normalise(Int_Wigner_WD))
#subplot(223)
#plot(t,normalise(Int_Wigner_TD_out))
#subplot(224)
#plot(Omega,normalise(Int_Wigner_WD_out))
#
#%%
#
#cfunXOmega=fit(Omega',E_Omega,'gauss1')
#% cfunXT=fit(t',E_XT_intX,'gauss1')
#y_test=exp(-((Omega)/(0.02775)).^2);
#figure_docked('test');clf; hold on
#plot(Omega,y_test)
#grid on
#xlim([-0.2 0.2])
#plot([-1 1], [0.5 0.5])
#plot([-1 1], [exp(-2) exp(-2)])
#
#%%
#E_X_out=normalise(trapz(abs(E_TD_Final).^2,1));
#E_T_out=normalise(trapz(abs(E_TD_Final).^2,2));
#
#s= fitoptions('Method','NonlinearLeastSquares',...
#    'Lower',[0, -0.05, 0],...
#    'Upper',[1, 0.05, 5],...
#    'Startpoint',[0.99, 0, 1],...
#    'MaxIter', 100000);
#
#myfittype = fittype(@( R2, x0, Sigma, x ) R2*exp(-(sqrt(2)*(x-x0)./(Sigma)).^2),'options',s);
#cfitX=fit(x',(E_X_out'),myfittype);
#C1_X=cfitX.Sigma;
#FWHM_X=C1_X/0.8493218;
#
#s2= fitoptions('Method','NonlinearLeastSquares',...
#    'Lower',[0, -0.05, 0],...
#    'Upper',[1, 0.05, 10000],...
#    'Startpoint',[0.99, 0, 2000],...
#    'MaxIter', 100000);
#myfittype2 = fittype(@( R2, x0, Sigma, x ) R2*exp(-(sqrt(2)*(x-x0)./(Sigma)).^2),'options',s2);
#
#cfitT=fit(t',E_T_out,myfittype2);
#C1_T=cfitT.Sigma;
#FWHM_T=C1_T/0.8493218;
#
#figure_docked('test');clf;hold on
#plot(t,E_T_out)
#plot(t,cfitT(t))
#%%
#
#figure_docked('Time_out_fourier_plane')
#colormap('jet')
#imagesc(x*1e2,t,abs(E_TD_Final).^2)
#title('Field at Fourier Plane-Time')
#xlabel('Waist (cm)')
#ylabel('Duration (fs)')
#axis tight
#axis square
#xlim([-2 2])
#set(gca,'box','on','linewidth',1.3,'fontsize',14)
#save2pdf('FOPA_FourierPlane1p8um_2D_Time_Space15f10')
#
#figure_docked('Omega_out_fourier_plane')
#colormap('jet')
#imagesc(x*1e2,Omega,abs(E_XOmega_out).^2)
#title(strcat('Output Field/\Delta\lambda=',num2str(DeltaLambdaFWHM),'nm'))
#% title(strcat('\Delta\lambda=',num2str(DeltaLambdaFWHM),'nm'))
#xlabel('Waist (cm)')
#ylabel('Pulsation (rad.fs-1)')
#axis tight
#axis square
#set(gca,'box','on','linewidth',1.3,'fontsize',14)
#xlim([-2 2])
#ylim([-0.5 0.5])
#save2pdf('FOPA_FourierPlane1p8um_2D_SpaceOmega15f10')
#%%
#figure_docked('Time_out_fourier_plane_X');clf; hold on
#plot(x*1e2,E_X_out,'linewidth',1.5)
#plot([-FWHM_X/2*100 -FWHM_X/2*100], [0 1],'linewidth',1.5)
#plot([FWHM_X/2*100 FWHM_X/2*100], [0 1],'linewidth',1.5)
#plot([-2 2], [0.5 0.5],'linewidth',1.5)
#title(strcat('FWHM=',num2str(FWHM_X*100),'cm'))
#xlabel('Waist (cm)')
#ylabel('Intensity (arb. units)')
#axis tight
#axis square
#grid on
#xlim([-2 2])
#set(gca,'box','on','linewidth',1.3,'fontsize',14)
#save2pdf('FOPA_FourierPlane1p8um_1D_Space')
#
#figure_docked('Time_out_fourier_plane_T');clf; hold on
#plot(t,E_T_out,'linewidth',1.5)
#plot([-FWHM_T/2 -FWHM_T/2], [0 1],'linewidth',1.5)
#plot([FWHM_T/2 FWHM_T/2], [0 1],'linewidth',1.5)
#plot([-5000 5000], [0.5 0.5],'linewidth',1.5)
#title(strcat('FWHM=',num2str(FWHM_T/1000),'ps'))
#xlabel('Duration (fs)')
#ylabel('Intensity (arb. units)')
#axis tight
#axis square
#grid on
#xlim([-FWHM_T FWHM_T])
#set(gca,'box','on','linewidth',1.3,'fontsize',14)
#save2pdf('FOPA_FourierPlane1p8um_1D_Time')
#
#%%
#E_X_out2=normalise(trapz(abs(E_XOmega_out).^2,1));
#E_Omega_out=normalise(smooth(trapz(abs(E_XOmega).^2,2),3));
#
#s3= fitoptions('Method','NonlinearLeastSquares',...
#    'Lower',[0, -0.05, 0],...
#    'Upper',[1, 0.05, 5],...
#    'Startpoint',[0.99, 0, 1],...
#    'MaxIter', 100000);
#
#myfittype3 = fittype(@( R2, x0, Sigma, x ) R2*exp(-(sqrt(2)*(x-x0)./(Sigma)).^2),'options',s3);
#cfitX2=fit(x',(E_X_out2'),myfittype3);
#C1_X2=cfitX2.Sigma;
#FWHM_X2=C1_X2/0.8493218;
#
#s4= fitoptions('Method','NonlinearLeastSquares',...
#    'Lower',[0.98, -0.05, 0],...
#    'Upper',[1, 0.05, 100],...
#    'Startpoint',[0.99, 0, 10],...
#    'MaxIter', 100000);
#myfittype4 = fittype(@( R2, x0, Sigma, x ) R2*exp(-(sqrt(2)*(x-x0)./(Sigma)).^2),'options',s4);
#
#cfitOmega=fit(Omega',E_Omega_out,myfittype4);
#C1_Omega=cfitOmega.Sigma;
#FWHM_Omega=C1_Omega/0.8493218;
#
#figure_docked('test');clf;hold on
#plot(Omega,smooth(E_Omega_out,2))
#plot(Omega,cfitOmega(Omega))
#% xlim([1200 2800])
#%%
#figure_docked('Omega_out_fourier_plane')
#colormap('jet')
#imagesc(x*1e2,Omega,abs(E_XOmega_out).^2)
#title(strcat('Output Field/\Delta\lambda=',num2str(DeltaLambdaFWHM),'nm'))
#% title(strcat('\Delta\lambda=',num2str(DeltaLambdaFWHM),'nm'))
#xlabel('Waist (cm)')
#ylabel('Pulsation (rad.fs-1)')
#axis tight
#axis square
#set(gca,'box','on','linewidth',1.3,'fontsize',14)
#xlim([-2 2])
#ylim([-0.5 0.5])
#save2pdf('FOPA_FourierPlane1p8um_2D_SpaceOmega')
#
#figure_docked('Omega_out_fourier_plane_X');clf; hold on
#plot(x*1e2,normalise(smooth(E_X_out2,3)),'linewidth',1.5)
#plot([-FWHM_X2/2*100 -FWHM_X2/2*100], [0 1],'linewidth',1.5)
#plot([FWHM_X2/2*100 FWHM_X2/2*100], [0 1],'linewidth',1.5)
#plot([-2 2], [0.5 0.5],'linewidth',1.5)
#title(strcat('FWHM=',num2str(FWHM_X2*100),'cm'))
#xlabel('Waist (cm)')
#ylabel('Intensity (arb. units)')
#axis tight
#axis square
#grid on
#xlim([-2 2])
#set(gca,'box','on','linewidth',1.3,'fontsize',14)
#save2pdf('FOPA_FourierPlane1p8um_1D_Space_fromOmega')
#
#figure_docked('Omega_out_fourier_plane_Omega');clf; hold on
#plot(Omega,E_Omega_out,'linewidth',1.5)
#plot([-FWHM_Omega/2 -FWHM_Omega/2], [0 1],'linewidth',1.5)
#plot([FWHM_Omega/2 FWHM_Omega/2], [0 1],'linewidth',1.5)
#plot([-5000 5000], [0.5 0.5],'linewidth',1.5)
#title(strcat('FWHM=',num2str(FWHM_Omega),'rad/fs'))
#xlabel('Pulsation (rad/fs)')
#ylabel('Intensity (arb. units)')
#axis tight
#axis square
#grid on
#xlim([-FWHM_Omega FWHM_Omega])
#set(gca,'box','on','linewidth',1.3,'fontsize',14)
#save2pdf('FOPA_FourierPlane1p8um_1D_Omega')
#
#%%
#G=gauss1D(t,0,7,1);
#ft_G=fftshift(fft(fftshift(G)));
#
#fittest=fit(Omega',abs(ft_G)','gauss1')
#
#FWHM_Omegatest=fittest.c1/0.8493218
#FWHM_lambdatest=FWHM_Omegatest/omega0*1800
#adzqsd=(1800e-9)^2*FWHM_Omegatest*1e15/2/pi/c*1e9
#figure_docked('test');clf
#plot(Omega,normalise(abs(ft_G).^2));hold on
#plot([-FWHM_Omegatest/2 -FWHM_Omegatest/2], [0 1])
#plot([+FWHM_Omegatest/2 +FWHM_Omegatest/2], [0 1])
#plot([-10 10],[0.5 0.5])
#xlim([-0.3 0.3])
#
#%%
#
#DeltaTfocus=(Lambda*1e3*2*w/(lines*c))/sqrt(1-(Lambda*1e3/2/lines).^2)
#DeltaLambdaFocus=(Lambda*1e3*lines/(pi*2*w))*2*log(2)*sqrt(1-(Lambda*1e3/2/lines).^2)