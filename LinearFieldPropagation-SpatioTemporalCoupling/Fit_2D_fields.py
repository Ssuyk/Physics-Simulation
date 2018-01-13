# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:12:39 2017

@author: Boulot
"""
import numpy as np
from scipy.optimize import fmin, fmin_bfgs, fmin_powell, differential_evolution, fmin_slsqp
import matplotlib.pyplot as plt
import Q_matrix as QM


def prepare_data(Data):
    Abs = np.abs(Data)
    Arg = np.angle(Data)
    return Abs, Arg    

def Abs_xt_func(x,t,Lambda,*args):
    """ 
    args[0] = E0 field amplitude
    args[1] = w0 beam waist
    args[2] = Tau Pulse duration
    args[3] = Pulse Front Tilt
    """
    
    return args[0]*np.exp(  -(  (x/args[1])**2+(t/args[2])**2 +2*args[3]*x*t )  ) 

def Abs_fit_func(x,*args):
    
    """ function specially made so that it respects fmin syntax, i.e 
    we search the best parameters x[i] so that f(x)-exp_data is minimize. 
    x[0] = E0 field amplitude
    x[1] = w0 beam waist
    x[2] = Tau Pulse duration
    x[3] = Pulse Front Tilt (as defined by Kostenbauder formalisme adapted by Arktur OE 2005)
    Here we just take the absolute value (as in sqrt(real**2+imag**2))
        
    Use the following function to fit : 
    xopt_abs = fmin(Abs_fit_func, p0_abs, args=(X,T,Exp_Abs),maxiter=1000, maxfun = 1000)
    p0_abs is first guess containing an initial value for the tuple [x[0],x[1],x[2],x[3]]
    Exp_Abs is experimental data
    """
    X_mat = args[0].ravel()
    T_mat = args[1].ravel()
    Exp_data = args[2].ravel()
    return np.linalg.norm(   Exp_data - x[0]*np.exp(  -(  ((X_mat-x[4])/x[1])**2+((T_mat-x[5])/x[2])**2 +2*x[3]*(X_mat-x[4])*(T_mat-x[5]) )  )   )
    
def fit_real_func(x,*args):
    
    """ function specially made so that it respects fmin syntax, i.e 
    we search the best parameters x[i] so that f(x)-exp_data is minimize.  

    x[0] = R wavefront curvature
    x[1] = Beta Chirp
    x[2] = Wave Front Rotation WFR
    """
    X_mat = args[0].ravel()
    T_mat = args[1].ravel()
    Lambda = args[2]
    Exp_data = args[3].ravel()
    x0 = args[4]
    t0 = args[5]
    return np.linalg.norm( Exp_data - np.cos((-Lambda/np.pi*(X_mat-x0)**2/x[0])+(x[1]*(T_mat-t0)**2)-2*x[2]*(X_mat-x0)*(T_mat-t0)))    
#    return np.linalg.norm( Exp_data -   E0*np.exp(  -(  (X_mat/w0)**2+(T_mat/Tau)**2 +2*PFT*X_mat*T_mat )  )*\
#                          np.cos((-Lambda/np.pi*X_mat**2/x[0])+(x[1]*T_mat**2)-2*x[2]*X_mat*T_mat))
        
def Fit_Exp_DataAbsRealImag(X,Y, Exp_Data, Lambda, p0_abs, p0_real, max_index):
    """
    We use previously defined function to fit our function, using fmin.
    Once the parameters are extracted, we arange them so that we can use the 
    defineQmatrix function without sorting the variable in the main file. 
    """
    """
    x[0] = E0 field amplitude
    x[1] = w0 beam waist
    x[2] = Tau Pulse duration
    x[3] = Pulse Front Tilt (as defined by Kostenbauder formalisme adapted by Arktur OE 2005)
    x[4] = R wavefront curvature
    x[5] = Beta Chirp
    x[6] = Wave Front Rotation WFR
    defineQmatrix(Lambda,x[4],x[1],x[2],x[5],x[3],x[6])
    """
    xopt_abs = fmin(Abs_fit_func, p0_abs, args=(X,Y,np.abs(Exp_Data)))
    print(xopt_abs)
    
    Npts = np.size(X[0,:])
    min_ind = int(Npts/2-max_index)
    max_ind = int(Npts/2+max_index)
    Temp = Exp_Data[min_ind:max_ind,min_ind:max_ind]
    real_exp = np.real(Temp)
    abs_exp = np.abs(Temp)
    X_temp = X[min_ind:max_ind,min_ind:max_ind]
    Y_temp = Y[min_ind:max_ind,min_ind:max_ind]
    x0 = xopt_abs[4]
    t0 = xopt_abs[5]
    xopt_real = fmin(fit_real_func, p0_real, args=(X_temp,Y_temp,Lambda,real_exp/abs_exp,x0,t0))#, maxiter = 1000, maxfun= 1000
    print(xopt_real)
    
    E0_fit = xopt_abs[0]
    Xopt_real = [xopt_real[0], xopt_abs[1], xopt_abs[2], xopt_real[1], xopt_abs[3], xopt_real[2]]

    return E0_fit, Xopt_real, x0, t0



def Plot_Fit_result(X,Y,Exp_Abs, Exp_Arg, Fit_Abs, Fit_Arg, Fig_number, xlabel, ylabel, colormap):
    fig = plt.figure(Fig_number)
    plt.clf()
    plt.subplot(2,2,1)
    plt.pcolormesh(X, Y, Exp_Abs**2, rasterized=True, cmap=colormap, vmin=0, vmax=np.max(Exp_Abs**2))
    plt.colorbar()
    plt.title('Exp. Abs.')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.subplot(2,2,2)
    plt.pcolormesh(X, Y, Fit_Abs**2, rasterized=True, cmap=colormap, vmin=0, vmax=np.max(Fit_Abs**2))
    plt.title('Fit Abs.')
    plt.colorbar()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.subplot(2,2,3)
    plt.pcolormesh(X, Y, Exp_Arg, rasterized=True, cmap=colormap)
    plt.colorbar()
    plt.title('Exp. Arg.')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.subplot(2,2,4)
    plt.pcolormesh(X, Y, Fit_Arg, rasterized=True, cmap=colormap)
    plt.title('Fit Arg.')
    plt.colorbar()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return fig
    
    
    
    
#%%    
## Test
##load data
#Temp_data = np.load('test.npy')
#
#t = np.real(Temp_data[1:,0])
#x = np.real(Temp_data[0,1:])
#[X,T] = np.meshgrid(x,t)
#
#Exp_Data = Temp_data[1:,1:]
#Exp_Abs, Exp_arg = prepare_exp_data(Exp_Data)
#
#
#p0_abs = [1,0.01,20,0] #initial guess
#xopt_abs = fmin(Abs_fit_func, p0_abs, args=(X,T,Exp_Abs),maxiter=1000, maxfun = 1000)
#print(xopt_abs)
#
#
#p0_arg = [1e5,0,1] #initial guess
#xopt_arg = fmin(Angle_fit_func, p0_arg, args=(X,T,1.9e-6,Exp_arg),maxiter=1000, maxfun = 1000)
#print(xopt_arg)
#
#y_out = Abs_xt_func(X,T,1.9,*xopt_abs)
#arg_out = Angle_xt_func(X,T,1.9,*xopt_arg)
#plt.figure(1)
#plt.clf()
#plt.subplot(1,2,1)
#plt.pcolormesh(X, T, Exp_Abs , rasterized=True, cmap='gnuplot2')
##plt.pcolormesh(X, T, Exp_arg , rasterized=True, cmap='gnuplot2')
#
#plt.title('exp')
#plt.subplot(1,2,2)
#plt.pcolormesh(X, T, y_out, rasterized=True, cmap='gnuplot2')
#
##plt.pcolormesh(X, T, arg_out, rasterized=True, cmap='gnuplot2')
#plt.title('fit')
