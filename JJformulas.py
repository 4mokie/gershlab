# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:28:59 2018

@author: kalas
"""


import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import numpy as np
import os

from tqdm import tqdm, tqdm_notebook
import math

from  scipy.special import digamma as ψ
import scipy.special as special

from mpmath import besseli as Iν
from scipy.constants import hbar, pi, h, e, k

from mpmath import mp

mp.dps = 25


kB = k
γ = 0.57721566
RQ = h/4/e**2
Φ0 = h/2/e  
Δ = 2.1*kB


def Cjj(A, metalTHK=250e-10, c_per_m2 = 50e-15*1e12):
    
    return c_per_m2 * (A + 2*np.sqrt(A)*metalTHK)

def Ec(C):
    
    return e**2/2/C/kB
    

def Qp(EJ, Ec, Rsh):

    return np.pi*Rsh/RQ*(EJ/2/Ec)**0.5


def Ic (R, Δ = 2.1*kB):
    return pi*Δ/2/e/R



def EJ_AB (R, Δ = 2.1*kB):
    
    Ic_ = Ic (R, Δ)
    
    return Ic_*Φ0/2/pi/kB


def EJ_star (EJ, R, T, C):
    β = 1/T/kB
    ωR = 1/R/C
    ρ = R/RQ
    
    return EJ*np.exp( -ρ*( ψ(1 + hbar*ωR*β/2/pi)+ γ) )


def EJ_star_simp (EJ, C):
    
    EC = e**2/C
    α = kB*EJ/EC/4
    
    return EJ*α/( 1+ α)

def  I_IZ( Vb, EJ, R, T):
    out = []

    β = 1/T/kB
    ρ = R/RQ
    Z = 1j*β*e*Vb/pi/ρ

    
    for z in Z:
       
        try :
            out = np.append(out, 2*e/hbar*EJ*kB * ( float((Iν(1-z, EJ/T) / Iν(-z, EJ/T)).imag) ))
        except OverflowError:
            print('¯\_(ツ)_/¯')
            out = np.append(out, 0 )

    return out


def  I_IZ_simp( Vb, EJ, R, T):
    out = []

    I0 = 2*e*EJ*kB/hbar

    Ω = 2*e*I0*R/hbar
    D = T*kB*R*(2*e/hbar)**2
    z = Ω/D
   

    return I0*z/2 * 2*e*Vb*D/hbar / ( (2*e*Vb/hbar)**2 + D**2  )

def find_R0_Isw( EJ, R_env , T, VERBOSE = False):
    
    
    
    Ic0 = EJ/ (Φ0/2/pi/kB)
    Vc0 = R_env*Ic0
    
    Vs = np.linspace(0, 2*Vc0, 201)
    
    Is = I_IZ( Vs, EJ = EJ, R = R_env, T = T)
    
    Is_max = np.max (Is)
    R0 = np.mean( (np.diff( Vs - 1*Is*R_env)/np.diff(Is))[:11] ) + 1 
    
    if VERBOSE:
        fig, ax = plt.subplots()
        
        ax.plot(Vs - 1*Is*R_env, Is)
        ax.axhline(Is_max, 0,1, ls = '--', label = r'$I_s = {:2.1f} nA$'.format(Is_max/1e-9))
        
        Iss = np.linspace (0, Ic0, 51)
        ax.plot( R0*Iss, Iss, label = r'$R_0 = {:2.3f} kOhm$'.format(R0/1e3) )
        
        ax.legend()
        
#         fig.close()
        print(Is_max)
        print(Ic0)
    
    return R0, Is_max

def find_Isw( RN, R_env , T, C ):

    Vs = np.linspace(0, 10e-3, 51)
    
#     EJ_s = EJ_star (EJ = EJ_AB(RN), R = R_env, T = T, C = C)
    EJ_s = EJ_star_simp (EJ = EJ_AB(RN),  C = C)
    
    
    Is = I_IZ( Vs, EJ = EJ_s, R = R_env, T = T) 

    return np.max(Is)


def find_R0( RN, R_env , T, C ):

    Vs = np.linspace(0, .1e-5, 51)
    
#     EJ_s = EJ_star (EJ = EJ_AB(RN), R = R_env, T = T, C = C)
    EJ_s = EJ_star_simp (EJ = EJ_AB(RN),  C = C)
    
    Is = I_IZ( Vs, EJ = EJ_s, R = R_env, T = T) 
    
    return np.mean(np.diff(Vs - Is*R_env)/np.diff(Is)) + 1


def  V_AH_star( I,  EJ, Rn,  T):
    Ic0 = EJ/( Φ0/2/pi/kB )
    i = I/Ic0
    Γ = 2*EJ/T
    i_ = (1 - i**2)**0.5
    
    return 2*Ic0*Rn* i_ * np.exp( -Γ*( i_ + i*np.arcsin(i) ))*np.sinh(np.pi/2*Γ*i)
    


def  V_AH( I,  EJ, Rn,  T):
    vs = []
    
    Ic0 = EJ/( Φ0/2/pi/kB )
    
    Γ = 2*EJ/T
    i_s = I/Ic0
    
    for i in i_s:
        if i < 0.95:
            i_ = (1 - i**2)**0.5
            v = 2*Ic0*Rn* i_ * np.exp( -Γ*( i_ + i*np.arcsin(i) ))*np.sinh(np.pi/2*Γ*i)
        elif i > 1.05:
            v = Ic0*Rn*(i**2 - 1)**0.5
        else:
            v = np.nan    
        vs.append(v)
    return np.array(vs)

def  II( R1, R2):
    return R1*R2/(R1+R2)

def  Iqp(  V, T, G1 = 1/6.76e3, G2 = 1/60e3, V0 = 0.15e-3 ):
    I = ( (G1-G2)*V0*np.tanh(V/V0) + G2*V)*np.exp(-Δ/T/kB)
    
    return I


def R0_IZ(EJ, R, T):
    
    return R/(Iν(0, EJ/T)**2 - 1 )


def Njump(i, Q, EJ, T):
    
    Γ = T/EJ
    
    z = (8/np.pi/Q/Γ)**0.5
    Ee = 8/np.pi**0.5/Q*( np.exp(-z**2)/z/special.erfc(z) - np.pi**0.5 ) 
    
    Ed = (1 + np.pi**2/8*Ee)**0.5
    im = np.pi/4*i*Q
    
    Np = 1 + 2*Q/np.pi**2*( Ed - 1 ) + i*Q**2/2/np.pi*np.log( ( Ed - im )/(1 - im) )
    Nm = 1 + 2*Q/np.pi**2*( Ed - 1 ) - i*Q**2/2/np.pi*np.log( ( Ed + im )/(1 + im) )

    
    return Np, Nm


def wpK(EjK, EcK):
    return np.sqrt(8*EjK*EcK)

def ΔU(i, EJ):
    ΔUp = 2*EJ*( 1*(1-i**2)**0.5 + i*(np.arcsin(i) - np.pi/2) )
    ΔUm = 2*EJ*( 1*(1-i**2)**0.5 + i*(np.arcsin(i) + np.pi/2) )

    return ΔUp, ΔUm

def τ(i, EJ, Ec, T):
    
    ωa = wpK(EJ, Ec)*kB/hbar * (1 - i**2)**0.25
    
    ΔUp, ΔUm = ΔU(i, EJ)
    
    τp = 2*np.pi/ωa*np.exp( ΔUp/T )
    τm = 2*np.pi/ωa*np.exp( ΔUm/T )
    
    return τp, τm

def τQ(i, EJ, Ec, T):
    ωa = wpK(EJ, Ec)*kB/hbar * (1 - i**2)**0.25

    ΔUp, ΔUm = ΔU(i, EJ)
        
    aqp = ( 864*np.pi*ΔUp/wpK(EJ,Ec) )**0.5
    aqm = ( 864*np.pi*ΔUm/wpK(EJ,Ec) )**0.5
    
    τQp = 2*np.pi/ωa /aqp  *np.exp(7.2* ΔUp / wpK(EJ,Ec) ) 
    τQm = 2*np.pi/ωa /aqm  *np.exp(7.2* ΔUm / wpK(EJ,Ec) )
    return  τQp, τQm


def V_KM(i, EJ, Ec, Q, T):
    
#     out = [np.nan for i in I]
    
    Ic0 = EJ/( Φ0/2/pi/kB )
    
   
    
    τp, τm = τ(i, EJ, Ec, T)
    Np, Nm = Njump(i, Q, EJ, T)
    
    τQp, τQm = τQ(i, EJ, Ec, T)
    
#     out = h/2/e*(Np/τp +1/τQp - Nm/τm - 1/τQm)
    out = h/2/e*(Np/τp  - Nm/τm)
    
#     out[np.where (abs(i) > 4/np.pi/Q) ] = np.nan 
    
    return out
    

def R0_KM(EJ, Ec, Q, T):    
    di = 0.01
    Ic0 = EJ/(Φ0/2/pi/kB)
    R0 = V_KM(di, EJ, Ec, Q, T)/di/Ic0
    
    return R0


