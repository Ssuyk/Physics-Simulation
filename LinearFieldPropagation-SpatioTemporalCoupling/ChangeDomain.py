# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:37:08 2017

@author: Boulot
"""

import numpy as np

def XT_2_XOmega(Qm1):

    y=np.zeros((2,2),dtype=np.complex_)
    y[0,0]=Qm1[0,0]+Qm1[0,1]**2/Qm1[1,1]
    y[0,1]=-(1j/2)*(Qm1[0,1]/Qm1[1,1])
    y[1,0]=-y[0,1]
    y[1,1]=1/(4*Qm1[1,1])
    
    return y

def XOmega_2_XT(Mat):
    y=np.zeros((2,2),dtype=np.complex_)
    y[0,0] = Mat[0,0]-Mat[0,1]**2/Mat[1,1]
    y[0,1] = 1j/2*(Mat[0,1]/Mat[1,1])
    y[1,0] = -y[0,1]
    y[1,1] = 1/(4*Mat[1,1])
    return y

def XT_2_KOmega(Qm1,Lambda):

    temp=XT_2_XOmega(Qm1)
    y=np.zeros((2,2),dtype=np.complex_)
    y[0,0]=1/(4*temp[0,0])
    y[0,1]=1j/2*(temp[0,1]/temp[0,0])
    y[1,0]=-y[0,1]
    y[1,1]=temp[1,1]+(temp[0,1]**2/temp[0,0])
    
    return y 

def XT_2_KT(Qm1, Lambda):

    temp=XT_2_KOmega(Qm1,Lambda)
    y=np.zeros((2,2),dtype=np.complex_)
    y[0,0]=temp[0,0]+temp[0,1]**2/temp[1,1]
    y[0,1]=(1j/2)*(temp[0,1]/temp[1,1])
    y[1,0]=-y[0,1]
    y[1,1]=(1/4)/temp[1,1]
     
    return y 