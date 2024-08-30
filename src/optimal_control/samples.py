# A set of utilities for generating uniform samples
# Author : Amit Parag
# Date : 09/03/2022 Toulouse



import numpy as np 
import pinocchio as pin
from optimal_control.utils import IK_position



def uniform_samples(nb_samples:int,robot,config,eps_q=0.9,eps_v=0.01):
    '''
    Samples initial states x = (q,v) within conservative state range
    '''

    q0			= np.asarray(config['q0'])
    id_endeff	= robot.model.getFrameId('contact')
    samples 	= []
    q_des, _, _ = IK_position(robot, q0, id_endeff, config['p_des'])


    nq = robot.model.nq 
    nv = robot.model.nv
    nx = nq + nv
    if nx < 14:
        """The reduced model"""
        q_lim = np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944])
        
    else:
        """The full model"""
        q_lim = np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])


    samples = []
    q_des, _, _ = IK_position(robot, q0, id_endeff, config['p_des'])
    q_min = np.maximum(q_des - eps_q*np.ones(nq), -q_lim) #0.9*np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])
    q_max = np.minimum(q_des + eps_q*np.ones(nq), +q_lim) #0.9*np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])
    v_min = np.zeros(nv) - eps_v*np.ones(nv) 
    v_max = np.zeros(nv) + eps_v*np.ones(nv) 
    x_min = np.concatenate([q_min, v_min])   
    x_max = np.concatenate([q_max, v_max])   
    for i in range(nb_samples):
        samples.append( np.random.uniform(low=x_min, high=+x_max, size=(nx,)))
    return np.array(samples)
