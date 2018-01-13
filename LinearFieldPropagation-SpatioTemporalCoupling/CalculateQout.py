# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.linalg import inv

def CalculateQoutfrom_K_and_Qtilde(Qtilde,K,Lambda):
    Qin = -1j*np.pi/(Lambda)*inv(Qtilde)
    A=K[0,0]
    B=K[0,1]  
    C=K[1,0]
    D=K[1,1]
    E=K[0,3]   
    F=K[1,3]
    G=K[2,0]    
    H=K[2,1]
    I=K[2,3]
    
    
    Alpha=np.matrix([[A, 0],[G, 1]])
    Beta=np.matrix([[B, E/(Lambda)],[H, I/(Lambda)]])

    Gamma=np.matrix([[C, 0],[0, 0]])
    Delta=np.matrix([[D, F/(Lambda)],[0, 1]])
    
    Qout=(Alpha*Qin+Beta)*inv(Gamma*Qin+Delta)
    return Qout


