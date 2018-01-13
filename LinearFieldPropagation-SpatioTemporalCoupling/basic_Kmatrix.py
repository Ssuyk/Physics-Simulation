# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:47:39 2017

@author: Boulot
"""

import numpy as np

def lens(f):
    Mat = np.matrix([[1, 0, 0, 0],[-1/f, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    return Mat

def Freespace(L,k2sec):
    Mat = np.matrix([[1, L, 0, 0],[0, 1, 0, 0],[0, 0, 1, 2*np.pi*L*k2sec],[0, 0, 0, 1]])
    return Mat

def grating(T_i,T_d, Lambda):
    c= 2.9997*10**(8) #m/s
    Mat = np.matrix([[-(np.cos(T_d)/np.cos(T_i)), 0, 0, 0],[0, -(np.cos(T_i)/np.cos(T_d)), 0, Lambda/c*(np.sin(T_d)-np.sin(T_i))/np.cos(T_d)*10**(15)]
    ,[1/c*(np.sin(T_i)-np.sin(T_d))/np.cos(T_i)*10**(15), 0, 1, 0],[0, 0, 0, 1]])
    return Mat


