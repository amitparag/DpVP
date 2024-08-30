
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


from optimal_control.utils import get_p
from optimal_control.utils import get_v

def plot_ddp_endeff(ddp_data,plot_warmstart=False,x=None,fig=None, ax=None, label=None, SHOW=True, marker=','):
    '''
    Plot ddp_data results (endeff)
    '''
    # Parameters
    dt = ddp_data['dt'] 
    nq = ddp_data['nq']

    if plot_warmstart:
        x = np.array(ddp_data['init_xs'])
    else:
        x = np.array(ddp_data['xs'])
        # Extract EE traj

    N = x.shape[0]-1
    q = x[:,:nq]
    v = x[:,nq:]
    p_EE = get_p(q, ddp_data['pin_model'], ddp_data['frame_id'])
    v_EE = get_v(q, v, ddp_data['pin_model'], ddp_data['frame_id'])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col', figsize=(11.6,3.71))
    #if(label is None):
    #    label='End-effector'
    xyz = ['x','y','z']
    for i in range(3):
        # Positions
        ax[i,0].plot(tspan, p_EE[:,i], linestyle='-', marker=marker, label=label)
        ax[i,0].set_ylabel('$P^{EE}_%s$'%xyz[i], fontsize='large')
        ax[i,0].grid(True)
        #Velocities
        ax[i,1].plot(tspan, v_EE[:,i], linestyle='-', marker=marker, label=label)
        ax[i,1].set_ylabel('$V^{EE}_%s$'%xyz[i], fontsize='large')
        ax[i,1].grid(True)
    ax[-1,0].set_xlabel('Time (s)', fontsize='x-large')
    ax[-1,1].set_xlabel('Time (s)', fontsize='x-large')
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    handles, labels = ax[i,0].get_legend_handles_labels()

    #if not label == None:
        #fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    plt.legend(bbox_to_anchor=(1.05,1.08),loc='upper right',fancybox=True, framealpha=0.5)
    plt.tight_layout()
    #fig.suptitle('End-effector positions and velocities', fontsize='xx-large')
    if(SHOW):
        plt.show()
    return fig, ax