# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:53:16 2019

@author: racco
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 17:45:43 2018 @author: wsLu
"""
from math import *
import scipy.constants as const
from scipy.special import ellipk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from si_prefix import si_format

roundN = 5

def JJparameters(RN, JJwidthUM=0.2, metalTHK=250E-10, Tc=1.34):
    
    Rs_JJ = RN * ( (JJwidthUM*1E-6 + 2*metalTHK) *JJwidthUM*1E-6 )
    IAB = const.pi *1.764 *const.k *Tc /2 /const.e /RN 
    EJ_JJ = const.h /2 /const.e /2 /const.pi *IAB
    C_JJ = 50E-15 *JJwidthUM *JJwidthUM 
    EC_JJ = const.e *const.e /2 /C_JJ
    C0 = ParplateCap(area= 56*1E-12 , dielecTHK=10E-9, epsilon = 9.34*const.epsilon_0)
    C0 = 1E-99
    EC0 = const.e *const.e /2 /C0
    
    freqPlasma = sqrt(2 *const.e *IAB *2*const.pi /const.h /max(C0,C_JJ)) /2/const.pi
    Q = freqPlasma*2*pi *RN *max(C0,C_JJ)
#    ParplateCap(area= JJwidthUM*JJwidthUM*1E-12 , dielecTHK=1E-9, epsilon = 9.34*8.828E-12)
#    return [IAB, EJ]
    return "[RN_JJ (kohm), Rs_JJ (kohm-um^2)] = " + format([round(RN/1E3, roundN), round(Rs_JJ/1E3/1E-12, roundN)]) +            "\n[I_AB (nA), EJ (mK)] = "            + format([round(IAB/1E-9, roundN), round(EJ_JJ/1E-3/const.k, roundN)]) +            "\n[C_JJQP (fF), EC (mK)] = "          + format([round(C_JJ/1E-15, roundN), round(EC_JJ/1E-3/const.k, roundN)]) +            "\n[C0 (fF), EC0 (mK)] = "             + format([round(C0/1E-15, roundN), round(EC0/1E-3/const.k, roundN)]) +            "\n[Freq_plasma (GHz))] = "            + format([round(freqPlasma/1E9, roundN)]) +            "\n[Q, Beta, EJ/EC] = "                + format([round(Q, 1), round(Q*Q, 1), round(EJ_JJ/min(EC0,EC_JJ), 1)])
#    return RN, round(Rs_JJ/1E3/1E-12, roundN), IAB, EJ_JJ/const.k, round(C_JJ/1E-15, roundN), round(EC_JJ/1E-3/const.k, roundN), round(C0/1E-15, roundN), round(EC0/1E-3/const.k, roundN), round(freqPlasma/1E9, roundN), round(Q, 1), round(Q*Q, 1), round(EJ_JJ/min(EC0,EC_JJ), 1)

def JJpar(RN, JJwidthUM=0.2, metalTHK=250E-10, Tc=1.34):
   
    Rs_JJ = RN * ( (JJwidthUM*1E-6 + 2*metalTHK) *JJwidthUM*1E-6 )
    IAB = const.pi *1.764 *const.k *Tc /2 /const.e /RN
    EJ_JJ = const.h /2 /const.e /2 /const.pi *IAB
    C_JJ = 50E-15 *JJwidthUM *JJwidthUM
    EC_JJ = const.e *const.e /2 /C_JJ
    C0 = ParplateCap(area= 56*1E-12 , dielecTHK=10E-9, epsilon = 9.34*const.epsilon_0)
    EC0 = const.e *const.e /2 /C0
#     C0 = 1e-99
   
    freqPlasma = sqrt(2 *const.e *IAB *2*const.pi /const.h /C_JJ) /2/const.pi
    Q = freqPlasma*2*pi *RN *C_JJ
    EJoEc = EJ_JJ/EC_JJ

#     key =  [ 'RN_JJ',   'Rs_JJ', 'I_AB',   'EJ', 'C_JJQP',   'EC', 'C0', 'EC0', 'Freq_plasma', 'Q', 'Beta', 'EJ/EC']
#     unit=  [   'ohm', 'ohm-m^2',    'A',    'J',      'F',    'J',  'F',   'J',          'Hz',  '',     '',      '']
#     lst = [[     RN ,    Rs_JJ ,   IAB , EJ_JJ ,    C_JJ , EC_JJ ,  C0 ,  EC0 ,    freqPlasma,  Q ,   Q*Q ,  EJoEc ]]
    key =  [ 'RN_JJ',    'Rs_JJ', 'I_AB',           'EJ',  'C_JJQP',          'EC',   '$\omega_p$', 'Q','$\\beta$', 'EJ/EC']
    unit=  [  'kohm','kohm*um^2',   'nA',            'K',      'fF',           'K',          'GHz',  '',     '',      '']
    lst = [[ RN/1e3 , Rs_JJ*1e9 ,IAB*1e9, EJ_JJ/const.k , C_JJ*1e15, EC_JJ/const.k, freqPlasma/1e9,  Q ,   Q*Q ,  EJoEc ]]
    JJparDFM = pd.DataFrame( data = list(zip(*lst)),      index = key ).transpose()
    JJparUNI = pd.DataFrame( data = dict(zip(key, unit)), index = [0] )

    return JJparDFM,JJparUNI

def EJEC(Rs_JJ, JJwidthUM, metalTHKUM=250E-4, Tc=1.34):
    
    RN = Rs_JJ / ( (JJwidthUM + 2*metalTHKUM) *JJwidthUM )
    IAB = const.pi *1.764 *const.k *Tc /2 /const.e /RN 
    EJ_JJ = const.h /2 /const.e /2 /const.pi *IAB
    C_JJ = 50E-15 *JJwidthUM *JJwidthUM 
    EC_JJ = const.e *const.e /2 /C_JJ
    C0 = ParplateCap(area= 56*1E-12 , dielecTHK=10E-9, epsilon = 9.34*const.epsilon_0)
    EC0 = const.e *const.e /2 /C0
    
    freqPlasma = sqrt(2 *const.e *IAB *2*const.pi /const.h /max(C0,C_JJ)) /2/const.pi
    Q = freqPlasma*2*pi *RN *max(C0,C_JJ)
    return [EJ_JJ/const.k, EC_JJ/const.k]

def CfromIVCoffset(IVCoffset):
    """
    Extract C from IV curve V offset for high Ibias branch with V = IR +e/2C
    IVCoffset: V offset from linear fits of IVC at high Ibias branch (V)
    return: C per JJ (fF/JJ)
    """
    C = const.e /2 /IVCoffset *1E15 *30 /2
#    return C
    return "C_IVCoffset/JJ (fF/JJ) = " + format(round(C, roundN))

def ParplateCap(area, dielecTHK, epsilon = 9.34*const.epsilon_0):
    """
    """
    C = epsilon *area /dielecTHK
    return C
#    return "C_parallelPlate (fF) = " + format(C *1E15)

def tiltedWashboardU(EJKBT, IbiasArr, ax):

   phi = np.arange(-0.1*np.pi, 8*np.pi, 0.025*np.pi)
   UArr=[]
   for Ibias in IbiasArr:
       U = -EJKBT*const.k*np.cos(phi) - const.h /2/np.pi /2/const.e *Ibias *phi
       UArr.append(U)

   ax.set_title('Tilted washboard', fontsize=16, fontweight='bold')
   ax.set_xlabel('$\phi$ (pi)')
   ax.set_ylabel('U (K)')

   i = 0   
   for i,U in enumerate(UArr):
       ax.plot(phi/np.pi, U/const.k,label = 'I$_b$ =' + format(si_format(IbiasArr[i])) + 'A')

   ax.text(0.82, 0.94, 'E$_J$ =' + format(si_format(EJKBT)) + 'K\n', verticalalignment='bottom', horizontalalignment='left',transform=ax.transAxes,color='black', fontsize=12)
   ax.grid(True)   
   ax.legend()