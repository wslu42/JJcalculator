# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 17:45:43 2018 @author: wsLu
"""
import numpy as np
import scipy.constants as const
pi, k, e, h, phi0 = const.pi, const.k, const.e, const.h, const.physical_constants['magn. flux quantum'][0]
import pandas as pd
pd.set_option("display.precision", 3)
import matplotlib.pyplot as plt
plt.close('all')
from si_prefix import si_format
def si_forlst(lst):
    return [si_format(i) for i in lst]

#
def JJpar(RN=1e3, sizeX=0.2e-6, sizeY=0.2e-6, metalTHK=250e-10, 
          T=20e-3, Nser=1, Npar=1, C_shunt=1e-20,
          ezread=False, EunHz=False, material = 'Al'):
    
    TcdX = {'Al' : 1.34,
            'Nb' : 9.20,
            'Pb' : 0,
            'V'  : 4.0,
            'Sn' : 3.72,
            'Nb_Ono1987': 4.18}
    
    SCgap  = 1.764*k*TcdX[material]
    JJarea = sizeX*sizeY + (sizeX+sizeY)*metalTHK
    RN_JJ  = RN /Nser *Npar
    Rs_JJ  = RN_JJ * JJarea

    I_ABT  = pi /2/e *SCgap /RN_JJ *np.tanh(SCgap /2/k/T)
    EJ_JJ  = phi0 /2/pi *I_ABT
    LK_JJ  = phi0 /2/pi /I_ABT

    C_JJ   = 50e-15 *JJarea*1e12
    EC_JJ  = e**2 /2/C_JJ #gives EC;QP
    C_0    = C_shunt
    C_tot  = C_JJ + C_0
    EC_tot = e**2 /2/C_tot

    Z_JJ   = np.sqrt(LK_JJ/C_tot) 
    EJoC_JJ= EJ_JJ/EC_JJ
    EJoC_to= EJ_JJ/EC_tot

    omegaP = 1/np.sqrt(LK_JJ*C_tot)
    omegaP = omegaP/2/pi
    omegaRC= 1/RN_JJ/C_tot
    Q      = omegaP/omegaRC
    Q = np.sqrt(2*pi/phi0 *I_ABT *RN_JJ**2 *C_tot)

    if EunHz: u = h
    else:     u = k
    
    key = ['RN_JJ',      'Rs_JJ' , 'I_AB',     'EJ', 'LK_JJ', 'C_JJ',   'Z', 'EC_tot',   'w_p', 'Q','EJ/EC_to'] 
    unit= [  'ohm','$\Omega m^2$',    'A',      'K',     'H',    'F', 'ohm',      'K',    'Hz',  '',        ''] 
    lst = [ RN_JJ ,        Rs_JJ ,  I_ABT, EJ_JJ/u ,  LK_JJ ,  C_JJ , Z_JJ , EC_tot/u,  omegaP,  Q ,  EJoC_to ]
    
    if EunHz:
        unit[3] = 'Hz'
        unit[7] = 'Hz'
    JJparDFM = pd.DataFrame( data = list(zip(*[lst])),      index = key ).transpose()
    JJparUNI = pd.DataFrame( data = dict(zip(key, unit)), index = [0] )
    JJparUNI = dict(zip(key, unit))

    if ezread:
        l = []
        for i,v in JJparDFM.iloc[0].items():
            l+=['{}{}'.format(si_format(v),JJparUNI[i])]
        l = pd.Series(l)
        l.index = JJparDFM.columns.tolist()
        l = l.to_frame().transpose()
        return l
    else:
        return [JJparDFM,JJparUNI]

#
def CfromIVCoffset(IVCoffset):
    """
    Extract C from IV curve V offset for high Ibias branch with V = IR +e/2C
    IVCoffset: V offset from linear fits of IVC at high Ibias branch (V)
    return: C per JJ (fF/JJ)
    """
    C = const.e /2 /IVCoffset *1E15 *30 /2
    return "C_IVCoffset/JJ (fF/JJ) = " + format(round(C, roundN))

#
def ParplateCap(area, dielecTHK, epsilon = 9.34*const.epsilon_0):
    return epsilon *area /dielecTHK

#
def tiltedWashboardU(EJKBT, IbiasArr, ax):
    phi = np.arange(-0.1*np.pi, 8*np.pi, 0.025*np.pi)
    UArr=[]
    for Ibias in IbiasArr:
        U = -EJKBT*const.k*np.cos(phi) - const.h /2/np.pi /2/const.e *Ibias *phi
        UArr.append(U)
     
    ax.set_xlabel('$\phi$ (pi)')
    ax.set_ylabel('U (K)')

    i = 0   
    for i,U in enumerate(UArr):
        ax.plot(phi/np.pi, U/const.k,label = 'I$_b$ =' + format(si_format(IbiasArr[i])) + 'A')

#
# def QQstar(srclst):
#     JJplst = toJJplst(srclst)[0].transpose().drop('Device').transpose().astype(float)
#     srclst = srclst.transpose().drop('Device').drop('Mat.').drop('dsgn').transpose().astype(float)
    
#     Q      = JJplst['Q']

#     lst = [0]*len(srclst['R0ZF'])
#     for i,v in enumerate(srclst['R0ZF']):
#             if v < 1000:
#                 lst[i] = 0
#             else:
#                 lst[i] = v
#     R0 = lst
#     freqPlasma = JJplst['$\omega_p$']*1e9 /2/np.pi
#     C_JJ = JJplst['C_JJQP']*1e-15

#     Q_star = freqPlasma * R0/srclst['#ser']*srclst['#par'] *C_JJ 
    
#     return [Q,Q_star]