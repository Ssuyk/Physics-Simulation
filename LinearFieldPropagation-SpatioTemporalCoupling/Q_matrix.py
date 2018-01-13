# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:59:27 2017

@author: Boulot
"""
import numpy as np
from numpy.linalg import inv

def defineQmatrix(Lambda,R,w,Tau,Beta,PFT,WFR):

    Qminus1=np.zeros((2,2),dtype=np.complex_)
    Qminus1[0,0] = -1j*(Lambda)/np.pi/R-(1/w**2)
    Qminus1[1,1] =-1j*Beta+1/Tau**2
    Qminus1[0,1] = PFT - 1j*WFR
    Qminus1[1,0] = -Qminus1[0,1]

    Q=inv(Qminus1)
    return Q, Qminus1

def Create_Field(X, Y, E0, Qminus1,x0,y0):
    return E0*np.exp(Qminus1[0,0]*(X-x0)**2 + 2*Qminus1[0,1]*(X-x0)*(Y-y0)-Qminus1[1,1]*(Y-y0)**2)


def ExtractInfofromQm1matrix(Lambda,Qm1):

    R=1/np.real(Qm1[0,0])
    w=np.sqrt(-Lambda/np.pi/np.imag(Qm1[0, 0]))
    Tau=np.sqrt(Lambda/np.pi/np.imag(Qm1[1,1]))
    Beta=np.pi/Lambda*np.real(Qm1[1,1])

    return R,w,Tau,Beta

def invQin(Qin,Lambda):
    return -1j*np.pi/(Lambda)*inv(Qin)

