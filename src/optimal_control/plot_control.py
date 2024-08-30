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




def plot_ddp_control(ddp_data,plot_warmstart=False,u=None,fig=None, ax=None, label=None, SHOW=True, marker=','):
    '''
    Plot ddp_data results (control)
    '''
    dt = ddp_data['dt'] 
    nu = ddp_data['nu']
    nu = 5

    if plot_warmstart:
        u = np.array(ddp_data['init_us'])
    else:
        u = np.array(ddp_data['us'])

    N = u.shape[0]
    # Plots
    tspan = np.linspace(0, N*dt-dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nu, 1, sharex='col', figsize=(19.2,10.8))
    #if(label is None):
    #    label='Control'    
    for i in range(nu):
        # Positions
        ax[i].plot(tspan, u[:,i], linestyle='-', marker=marker,label=label)
        ax[i].set_ylabel(r'$u_%s$'%i, fontsize='x-large')
        ax[i].grid(True)
    ax[-1].set_xlabel('Time (s)', fontsize='x-large')
    fig.align_ylabels(ax[:])
    handles, labels = ax[i].get_legend_handles_labels()
    if not label == None:
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    #fig.suptitle('Control trajectories',fontsize='xx-large')
    if(SHOW):
        plt.show()
    return fig, ax