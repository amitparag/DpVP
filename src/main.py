
# The main iterative training loop
# Author: Amit Parag
# Date : 11 May, 2022

import numpy as np
import torch
import yaml
import pickle




from datagen import solve_ocps
from datagen import sampling_strategy
from train import learn_value,learn_policy
from nn import ActorResidualCritic
from optimal_control.samples import uniform_samples
from optimal_control.robot import load_robot
from optimal_control.path import exp_path


DTYPE = torch.float64
torch.set_default_dtype(DTYPE)



robot, config = load_robot()



def create_info(datas):

    x0s,v,vx,xs,us,iters_taken,sc    =   datas
    
    info = {}
    info["xs"]      =   xs
    info["iters"]   =   iters_taken
    info["sc"]      =   sc
    info["x0s"]     =   x0s
    info["us"]      =   us
    info["v"]       =   v
    return info

def differential_value_programming(dvp_params_dict,exp_number):




    params      =   dvp_params_dict[f'exp_{exp_number}']

    savepath    =   exp_path(exp_number=exp_number,make_path=True,get_path=True)

    train_v_params      =   params['learning_value']
    train_xs_params     =   params['learning_xs']
    train_us_params     =   params['learning_us']
    datagen_ocp_params  =   params['datagen_params']


    nb_samples          =   datagen_ocp_params['nb_samples']
    horizon             =   datagen_ocp_params['horizon']

    episodes            =   params["dvp_params"]["episodes"]
    max_action          =   params["dvp_params"]["max_action"]

    datagen_info        =   {}










    print("\t\t\t\t\t\t\t\tDVP ITERATION 1  \n")

    x0s                                 = uniform_samples(nb_samples=nb_samples,robot=robot,config=config)
    
    [x0s,v,vx,xs,us,iters_taken, sc]    = solve_ocps(x0s=x0s,horizon=horizon,robot=robot,config=config)
    model                               = ActorResidualCritic(x0s_dims=x0s.shape[1],
                                                            us_dims=us.shape[1]//horizon,
                                                            horizon=horizon,
                                                            max_action=max_action
                                                            )

    print("\n")

    datas       = [x0s,v,vx,xs,us]
    info_us, model  = learn_policy(policy='control',training_params=train_us_params, datas=[x0s,us], model=model)     ### Learing Policy
    info_xs, model  = learn_policy(policy='state',training_params=train_xs_params, datas=[x0s,xs],   model=model)     ### Learning state 
    info_v, model   = learn_value(training_params=train_v_params, datas=[x0s,v,vx],  model=model)


    datagen_info["eps_1"]    =  {'train_us':info_us,
                                'train_xs':info_xs,
                                'train_v':info_v,
                                "datagen":create_info([x0s,v,vx,xs,us,iters_taken, sc])}
       

    info_us     =   None
    info_xs     =   None

    print("\n\n")
    torch.save(model,savepath+f"/eps_1.pth")


    #### learning value function here
    for i in range(1,episodes):

        print("\n")
        print(f"\t\t\t\t\t\t\t\tDVP ITERATION {i+1}  \n")

        x0s                                     =   sampling_strategy(robot=robot,config=config,nb_samples=nb_samples,xs=xs)
        [x0s,v,vx,xs,us,iters_taken, sc]        =   solve_ocps(x0s=x0s,horizon=horizon,actorResidualCritic=model,robot=robot,config=config,use_warmstart=True)

        info_v, model                           =   learn_value(training_params=train_v_params, datas=[x0s,v,vx], model=model)
        datagen_info[f"eps_{i+1}"]              =   {'train_v':info_v,
                                                    "datagen":create_info([x0s,v,vx,xs,us,iters_taken, sc])}
        
        torch.save(model,savepath+f"/eps_{i+1}.pth")
        ### Free Memory
        us = None
        iters_taken = None
        sc = None

    ### 
    ### So refine policy after the terminal value iterative loop is complete to save on computation

    xs = None
    x0s                                             =   sampling_strategy(robot=robot,config=config,nb_samples=nb_samples,xs=None)
    [x0s,v,vx,xs,us,iters_taken, sc]                =   solve_ocps(x0s=x0s,horizon=horizon,actorResidualCritic=model,robot=robot,config=config,use_warmstart=True)
    
    ## Refining learned polcies
    info_us, model  = learn_policy(policy='control',training_params=train_us_params, datas=[x0s,us], model=model)     ### Learing Policy
    info_xs, model  = learn_policy(policy='state',training_params=train_xs_params, datas=[x0s,xs],   model=model)     ### Learning state 
    
    
    
    datagen_info[f"eps_{episodes+1}"]   =   {'train_xs':info_xs,
                                            'train_us':info_us,
                                            "datagen":create_info([x0s,v,vx,xs,us,iters_taken, sc])}
    
    
    
    torch.save(model,savepath+f"/eps_{episodes+1}.pth")

    
    
    logs = savepath+f'/logs.pkl'
    with open(logs, "wb") as f:     ### To store logs
        pickle.dump(datagen_info, f)

    f.close()    



if __name__=='__main__':



    with open('exp.yml') as f:

        dvp = yaml.load(f)
        print("\n\n")

    exp_number = 1
    print(f"\n\t\t\t\t\t\t\t\tExperiment {exp_number}")
    differential_value_programming(dvp,exp_number=exp_number)







