import numpy as np
import matplotlib.pyplot as plt

from wavefunction1d import *


def get_CPB_spec(Ej, Ec, ng_list):




    NQ = 9

    Nng = 101

    Q = np.arange(-NQ, NQ +1)

    Q_mtx = np.diag( np.ones( len(Q) ) )

    J_mtx = np.diag( np.ones(2*NQ) , 1) + np.diag( np.ones(2*NQ) , -1)


    Nfi = 301
    fi_lim = 3*2*np.pi

    fi = np.linspace(-fi_lim, fi_lim, Nfi)

    E = np.zeros( (Nng, 2*NQ+1))
    ##               (ng,  band,    charge/phase)
    Psi = np.zeros( (Nng, 2*NQ+1, 2*NQ+1) )
    Psi_fi = np.zeros( (Nng, 2*NQ+1 , Nfi ) )




    for i, ng in enumerate(ng_list):

        H = 4*Ec*(Q - ng)**2 * Q_mtx - Ej/2 * J_mtx 

        evals, evecs = solve_eigenproblem(H)

        E[i] = evals
        Psi[i] =  evecs
        
        
    return E


def plot_Ejcos(Ej, Ec, N):
    
    ng_list = np.linspace(-1, 1, 101)

    fig, ax = plt.subplots()
    ax.plot(ng_list, Ej*np.cos(ng_list*2*np.pi))
    
    Es = get_CPB_spec(Ej = Ej, Ec = Ec, ng_list = ng_list)
    
    for i in range(N):
        
        Emax, Emin = max(Es[:,i]), min(Es[:,i])
        ax.fill_between(ng_list, Emax, Emin, alpha = 0.3)
        ax.hlines(Emax,-1,1, alpha = 0.3, color = f'C{i}')
        ax.hlines(Emin,-1,1, alpha = 0.3, color = f'C{i}')
    

    wp = np.sqrt(8*Ej*Ec)
    
    ax.set_title('Ej = {:1.2f}, Ec = {:1.2f}, wp = {:1.2f}'.format(Ej,Ec,wp) )
    print('Ej = {:1.2f}, Ec = {:1.2f}, wp = {:1.2f}'.format(Ej,Ec,wp))
    return ax



    