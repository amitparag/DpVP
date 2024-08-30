# A set of utilities for plotting
# Author : Amit Parag
# Date : 09/03/2022 Toulouse


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

sns.set(font_scale=1.5)
plt.rc('text', usetex=True)

sns.set_context("paper",rc={"lines.linewidth": 2.5})
sns.axes_style("white")
sns.color_palette("husl", 3)
sns.set(font_scale = 1.5)
sns.set_style("whitegrid", {'axes.grid' :False,'grid.linestyle': '--'})






def plot_ddp_state(ddp_data,plot_warmstart=False,fig=None, ax=None, label=None, SHOW=True, marker=','):
    '''
    Plot ddp_data results (state)
    '''
    # Parameters
    dt = ddp_data['dt'] 
    nq = ddp_data['nq']
    nv = ddp_data['nv']
    nq = 6

    if plot_warmstart:
        x = np.array(ddp_data['init_xs'])
    else:
        x = np.array(ddp_data['xs'])

    N  = x.shape[0]-1


        # Extract pos, vel trajs
    q = x[:,:nq]
    v = x[:,nv:]
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nq, 2, sharex='col', figsize=(19.2,10.8))
    #if(label is None):
    #    label='State'
    for i in range(nq):
        # Positions
        ax[i,0].plot(tspan, q[:,i], linestyle='-', marker=marker, label=label)
        ax[i,0].set_ylabel(r'$q_%s$'%i, fontsize='x-large')
        ax[i,0].grid(True)
        # Velocities
        ax[i,1].plot(tspan, v[:,i], linestyle='-', marker=marker, label=label)
        ax[i,1].set_ylabel(r'$v_%s$'%i, fontsize='x-large')
        ax[i,1].grid(True)


    ax[-1,0].set_xlabel('Time (s)', fontsize='x-large')
    ax[-1,1].set_xlabel('Time (s)', fontsize='x-large')
    fig.align_ylabels(ax[:, 0])
    fig.align_ylabels(ax[:, 1])
    # Legend
    handles, labels = ax[i,0].get_legend_handles_labels()
    if not label == None:
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    #fig.suptitle('State : joint positions and velocities', fontsize='xx-large')
    if(SHOW):
        plt.show()
    return fig, ax


def plot_xs():
    pass