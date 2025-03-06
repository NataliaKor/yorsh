# This code is based on https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/blob/Kerr_Equatorial_Eccentric/integrate_backwards.py

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from few.trajectory.inspiral import EMRIInspiral
from few.waveform import (
    GenerateEMRIWaveform,
)

from few.utils.constants import *

from few.utils.utility import (
    get_separatrix,
    get_mu_at_t,
    get_p_at_t,
)


def get_e0p0(M, mu, e_f, Phi_phi0, Phi_theta0, Phi_r0, T, dt, dist, a, x0):
    
    p_f = get_separatrix(a, e_f, x0) + 0.1 # hasto add a bit for the code to work
    print('e_f = ', e_f)
    print('p_f = ', p_f)
    print('p_s = ', 6. + 2.*e_f)
    trajectory_class = "SchwarzEccFlux"
    
    inspiral_kwargs_back = {
        "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # All of the trajectories will be well under len = 1000
        "err": 1e-12,  # Set error tolerance on integrator -- RK8
        "integrate_backwards": True,  # Integrate trajectories backwards
    }
   
    inspiral_kwargs_forward = {
        "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # All of the trajectories will be well under len = 1000
        "err": 1e-12,  # Set error tolerance on integrator -- RK8
        "integrate_backwards": False,  # Integrate trajectories backwards
    }

    # Set up trajectory module for backwards integration
    traj_backwards = EMRIInspiral(func=trajectory_class)
    
    # Generate backwards integration
    t_back, p0, e0, x0, Phi_phi_back, Phi_r_back, Phi_theta_back = (
        traj_backwards(
            M,
            mu,
            a,
            p_f,
            e_f,
            x0,
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            dt=dt,
            T=T,
            **inspiral_kwargs_back
        )
    )

    print('p0 = ', p0)
    print('e0 = ', e0)
    print('t_back = ', t_back)
    
    # To double check p0
    #p0 = get_p_at_t(traj_backwards, T*0.999, [M, mu, 0.0, e0[-1], 1.]) #, bounds=[get_separatrix(a,e0,1.0)+0.1, 40.0])
 
    # Verify with the forward trajectory
    traj_forwards = EMRIInspiral(func=trajectory_class)
    t_forward, p_forward, e_forward, _, _, _, _ = (
        traj_forwards(
            M,
            mu,
            a,
            p0[-1],
            e0[-1],
            x0[-1],
            Phi_phi0=Phi_phi_back[-1],
            Phi_theta0=Phi_theta_back[-1],
            Phi_r0=Phi_r_back[-1],
            dt=dt,
            T=T,
            **inspiral_kwargs_forward
        )
    )
    print('p_forward = ', p_forward)
    print('e_forward = ', e_forward)
    print('t_forward = ', t_forward)
    # Make plots
    #fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    #ax1, ax2, ax3 = axes.flatten()
    #ax1.plot(range(10), 'r')
    #ax2.plot(range(10), 'b')
    #ax3.plot(range(10), 'g')
    #plt.savefig('trajectory.png')
    #plt.close()
    fig, axes = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.3)
    fig.set_size_inches(14, 8)
    axes = axes.ravel()
    ylabels = [r'$e$', r'$p$', r'$e$']
    xlabels = [r'$p$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$']
    ys = [e_forward, p_forward, e_forward]
    xs = [p_forward, t_forward, t_forward]

    for i, (ax, x, y, xlab, ylab) in enumerate(zip(axes, xs, ys, xlabels, ylabels)):
        ax.plot(x, y)
        ax.set_xlabel(xlab, fontsize=16)
        ax.set_ylabel(ylab, fontsize=16)
    plt.savefig('trajectory.png')
    return e0[-1], p0[-1]



