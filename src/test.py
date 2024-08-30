# Author: Amit Parag
# Date : 11 May, 2022


import numpy as np
import torch
from tqdm import tqdm
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from torch.nn.functional import mse_loss as mse
import pandas

from datagen import solve_ocps
from nn import ActorResidualCritic
from optimal_control.ddp import init_ddp,extract_ddp_data
from optimal_control.samples import uniform_samples
from optimal_control.robot import load_robot
from optimal_control.plot_results import plot_warmstart
from optimal_control.plot_results import plot_warmstarts
from optimal_control.path import exp_path



sns.set(font_scale=1.5)
plt.rc('text', usetex=True)

sns.set_context("paper",rc={"lines.linewidth": 2.5})
sns.axes_style("white")
sns.color_palette("husl", 3)
sns.set(font_scale = 1.5)
sns.set_style("whitegrid", {'axes.grid' :False,'grid.linestyle': '--'})

##################################################33









robot, config = load_robot()





def check_quality_of_policy(model):

    x0s         =   uniform_samples(nb_samples=20,robot=robot,config=config)
    idx         =   np.random.choice(range(0,20))

    ddp         =   init_ddp(robot=robot,config=config,x0=x0s[idx],N_h=500)
    ddp.solve()
    ddp_data    =   extract_ddp_data(ddp)
    xs, us      =   model.warmstart(x0s[idx])

    ddp_data["init_xs"] = np.array(xs)[0:101,:]
    ddp_data["init_us"] = np.array(us)[0:100,:]

    plot_warmstart(ddp_data=ddp_data)


def check_iters(model):
    x0s     = uniform_samples(nb_samples=10,robot=robot,config=config)

    for x0 in x0s:


        ddp     =   init_ddp(robot=robot,config=config,x0=x0,N_h=100)
        ddp.solve()
        
        xs, us  =   model.warmstart(x0)
        xs      =   xs[:101]
        us      =   us[:100]
        ddp2    =   init_ddp(robot=robot,config=config,x0=x0,N_h=100)
        ddp2.solve(init_xs=xs,init_us=us,maxiter=1000,isFeasible=False)
        


        print(ddp.iter,ddp2.iter)



def plot_many(model):

    x0s     = uniform_samples(nb_samples=10,robot=robot,config=config)

    ddp     =   init_ddp(robot=robot,config=config,x0=x0s[1],N_h=model.horizon)
    ddp.solve()

    data = extract_ddp_data(ddp)
    dt = data['dt'] 
    nq = data['nq']
    nv = data['nv']
    nu = data['nu']
    pm =  data['pin_model'] 
    fid = data['frame_id']  

    DDPS_DATAS = []

    for x0 in x0s:

        xs, us = model.warmstart(x0)
        ddp_data = {}
        ddp_data["xs"] = xs[0:101]
        ddp_data["us"] = us[0:100]
        ddp_data["dt"] = dt
        ddp_data['nq'] = nq
        ddp_data['nv'] = nv
        ddp_data['nu'] = nu
        ddp_data['pin_model'] = pm
        ddp_data['frame_id'] = fid


        DDPS_DATAS.append(ddp_data)

    plot_warmstarts(DDPS_DATAS,SHOW=True)


    


def check_iters(model,nb_samples=160):


    x0s     = uniform_samples(nb_samples=nb_samples,robot=robot,config=config)
    #print("Base\t\tResidual\t\tDisjoint")

    ddp_iter    =   []
    dvp_iter    =   []


    for x0 in tqdm(x0s):

        ### BASE
        ddp     =   init_ddp(robot=robot,config=config,x0=x0,N_h=100)
        ddp.solve()
        ddp_iter.append( ddp.iter )
        


        ### Disjoint
        xs, us  =   model.warmstart(x0)
        ddp3    =   init_ddp(robot=robot,config=config,x0=x0,actorResidualCritic=model,N_h=100)
        xs      =   xs[:101]
        us      =   us[:100]
        ddp3.solve(init_xs=xs,init_us=us,maxiter=1000,isFeasible=False)
        dvp_iter.append( ddp3.iter )
        

    avg_ddp_iters  = sum(ddp_iter)/len(ddp_iter)
    avg_dvp_iters  = sum(dvp_iter)/len(dvp_iter)


    fig, ax = plt.subplots(figsize=(11.6,3.71))

    ax.scatter(y=np.arange(0,len(ddp_iter)), x=ddp_iter,marker=".",s=50,label='DDP',vmax=20,c='teal')
    ax.scatter(y=np.arange(0,len(dvp_iter)), x=dvp_iter,marker=".",s=50,label=r"$\partial$PVP",vmax=20,c='darkorchid')
    
    ax.set_ylabel("OC Problem Number")
    ax.set_xlabel("Rollouts required",labelpad=10)
    ax.set_xticks(np.arange(0,max(ddp_iter),step=2))
    ax.set_yticks(np.arange(0,len(ddp_iter)+1,step=200))


    ax.grid(True)
    plt.legend()
    print(f"Avg DDP iters : {avg_ddp_iters}")
    print(f"Avg DPvP iters : {avg_dvp_iters}")
    plt.tight_layout()

    plt.show()



def convergence_relative_criteria(save_path,eps_final=21):

    x0s         =   uniform_samples(nb_samples=50,robot=robot,config=config)
    x0s         =   torch.tensor(x0s,requires_grad=True)
    diff        =   []

    for i in np.arange(1,eps_final-1,step=1):
        model1       =   torch.load(save_path+f'/eps_{i}.pth')
        model2       =   torch.load(save_path+f'/eps_{i+1}.pth')

        v1,_ = model1(x=x0s,state=False,control=False,value=True)
        v2,_ = model2(x=x0s,state=False,control=False,value=True)

        error = mse(v1,v2)
        diff.append(error.detach().numpy().item())

    #diff = diff[10:]

    plt.figure(figsize=(19.8,5.2))
    plt.plot(diff,c='purple',linewidth=5)
    plt.scatter(x = range(0,len(diff)),y=diff,marker="|",s=500,c='red')
    plt.yscale('log')
    plt.ylim([10**-5,10**-2])
    plt.xlim([-1,len(diff)])
    plt.grid()

    xlabels = []
    for idx in range(len(diff)):
        label = f"{idx+1}:{idx+2}"
        xlabels.append(label)
    plt.xticks(ticks=range(len(diff)),labels=xlabels)
    plt.ylabel("MSE",fontsize='xx-large',rotation='horizontal',loc='top')
    plt.xlabel('Successive Value Functions',fontsize = 'x-large',labelpad=10)
    plt.tight_layout()
    plt.show()



def check_logs(save_path):
    
    logs_path = save_path + '/logs.pkl'
    logs = pandas.read_pickle(logs_path)
    rollouts = []
    train_time = 0

    for eps in range(1,21):
        rollouts.append([item for item in logs[f"eps_{eps}"]["datagen"]["iters"]])
        train_time += logs[f"eps_{eps}"]["train_v"]['Train Time']

    for eps in [1,21]:
        train_time += logs[f"eps_{eps}"]["train_xs"]['Train Time']
        train_time += logs[f"eps_{eps}"]["train_us"]['Train Time']


    rollouts = sum(rollouts, [])

    print("Total rollouts requires during training",sum(rollouts))
    print("Total training time",train_time)





if __name__=='__main__':

    torch.manual_seed(0)
    np.random.seed(0)

    exp_number  =   1
    eps_number  =   21
    save_path   =   exp_path(exp_number=exp_number,make_path=False,get_path=True)
    model       =   torch.load(save_path+f'/eps_{eps_number}.pth')
    

    check_quality_of_policy(model)
    check_iters(model,600)
    convergence_relative_criteria(save_path=save_path,eps_final=21)
    check_logs(save_path)
