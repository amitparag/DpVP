# A set of utilities for plotting
# Author : Amit Parag
# Date : 09/03/2022 Toulouse


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib
import matplotlib.backends.backend_pdf



from optimal_control.plot_states import plot_ddp_state
from optimal_control.plot_control import plot_ddp_control
from optimal_control.plot_ee import plot_ddp_endeff


def plot_warmstart(ddp_data,savePath=None):
    """
    Takes the network and plots the state
    """

    fig, ax = plt.subplots(3, 2, sharex='col',figsize=(19.2,10.8))


    #fig_xs,ax_xs = plot_ddp_state(ddp_data=ddp_data,plot_warmstart=True,SHOW=False,marker='o',label="Predicted State Trajectory")
    #fig_us,ax_us = plot_ddp_control(ddp_data=ddp_data,plot_warmstart=True,SHOW=False,marker='o',label="Predicted Control Trajectory")
    fig_eff,ax_eff = plot_ddp_endeff(ddp_data=ddp_data,plot_warmstart=True,SHOW=False,marker='o',label="Predicted State Trajectory: End-Eff")

    ### DDP INDefinite horizon
    #fig_xs,ax_xs = plot_ddp_state(ddp_data=ddp_data,plot_warmstart=False,fig=fig_xs,ax=ax_xs,SHOW=False,marker='o',label="Ground Truth")
    #fig_us,ax_us = plot_ddp_control(ddp_data=ddp_data,plot_warmstart=False,fig=fig_us,ax=ax_us,SHOW=False,marker='o',label="Ground Truth")
    fig_eff,ax_eff = plot_ddp_endeff(ddp_data=ddp_data,plot_warmstart=False,fig=fig_eff,ax=ax_eff,SHOW=False,marker='o',label="Quasi Steady State Reference")




    """
    if savePath is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(savePath+f".pdf")
        for fig in [fig_xs,fig_us,fig_eff]:
            pdf.savefig( fig )
        pdf.close()
        plt.close('all')
    
    else:
        plt.show()
    """
    plt.show()


def plot_warmstarts(DDPS_DATA, which_plots='all', labels=None, SHOW=False, marker=None, sampling_plot=1):
    '''
    Plot ddp results from 1 or several DDP solvers
    X, U, EE trajs
    INPUT 
      DDPS_DATA    : DDP solver data or list of ddp solvers data
      robot       : pinocchio robot wrapper
      name_endeff : name of end-effector (in pin model) 
    '''
    if(type(DDPS_DATA) != list):
        DDPS_DATA = [DDPS_DATA]
    if(labels==None):
        labels=[None for k in range(len(DDPS_DATA))]
    for k,d in enumerate(DDPS_DATA):
        # Return figs and axes object in case need to overlay new plots
        if(k==0):
            if('x' in which_plots or which_plots =='all'):
                fig_x, ax_x = plot_ddp_state(DDPS_DATA[k], label=labels[k], SHOW=False, marker=marker)
            if('u' in which_plots or which_plots =='all'):
                fig_u, ax_u = plot_ddp_control(DDPS_DATA[k], label=labels[k], SHOW=False, marker=marker)
            if('p' in which_plots or which_plots =='all'):
                fig_p, ax_p = plot_ddp_endeff(DDPS_DATA[k], label=labels[k], SHOW=False, marker=marker)

        # Overlay on top of first plot
        else:
            if(k%sampling_plot==0):
                if('x' in which_plots or which_plots =='all'):
                    plot_ddp_state(DDPS_DATA[k], fig=fig_x, ax=ax_x, label=labels[k], SHOW=False, marker=marker)
                if('u' in which_plots or which_plots =='all'):
                    plot_ddp_control(DDPS_DATA[k], fig=fig_u, ax=ax_u, label=labels[k], SHOW=False, marker=marker)
                if('p' in which_plots or which_plots =='all'):
                    plot_ddp_endeff(DDPS_DATA[k], fig=fig_p, ax=ax_p, label=labels[k], SHOW=False, marker=marker)

    if(SHOW):
      plt.show()
    
    fig = {}
    fig['p'] = fig_p
    fig['x'] = fig_x
    fig['u'] = fig_u

    ax = {}
    ax['p'] = ax_p
    ax['x'] = ax_x
    ax['u'] = ax_u

    return fig, ax

 
