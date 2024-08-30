# This script generates datasets used for training
# Author: Amit Parag
# Date : 11 May, 2022




from tqdm import trange
import numpy as np

from optimal_control.ddp import init_ddp
from optimal_control.samples import uniform_samples
from optimal_control.utils import get_u_grav


def sampling_strategy(robot,config,nb_samples,xs=None):

    """
    Either new samples uniformly sampled when learning v, xs, us
    or when learning v, samples from xs[-1]


    """
    if xs is None:
        return uniform_samples(nb_samples=nb_samples,robot=robot,config=config)


    else:


        refine_samples = uniform_samples(nb_samples=int((nb_samples//4)),robot=robot,config=config)


        if xs.ndim != 3:
            xs = xs.reshape(xs.shape[0],-1,refine_samples.shape[1])

        np.random.shuffle(xs)

        xs = xs[:int(3*(nb_samples//4))]

        for trajectory in xs:

            """
            The size of the generated dataset is bigger the nb_samples.
            Either comment out the next two lines
            or randomly return the nb_samples from the refine_samples array

            """

            #refine_samples =    np.vstack((refine_samples,trajectory[1]))
            refine_samples =    np.vstack((refine_samples,trajectory[trajectory.shape[0]//2]))
            refine_samples =    np.vstack((refine_samples,trajectory[trajectory.shape[0]//4]))


        
        np.random.shuffle(refine_samples)


        return refine_samples[:nb_samples] 








def outlier_check(ddp):
  """
  Takes a ddp object and checks for irrelarities
  
  Returns bool : Outlier or not
  """
  outlier = True if (ddp.cost > 1.0) else False
  return outlier






def solve_ocps(x0s,horizon,robot,config,actorResidualCritic=None,use_warmstart=False):
    
    """datagen for every episode"""

    print("\n")
    if actorResidualCritic is not None:
        actorResidualCritic.to('cpu')

    init_x0s,v,vx,xs,us  =   [],[],[],[],[]
    
    iters_taken         =   []
    stopping_criteria   =   [] 


    t_bar       =   trange(len(x0s), desc = " Solving OCPS ... ")

    for x0,_ in zip(x0s,t_bar):

        #ddp.problem.x0 = x0
        nq      =   robot.model.nq
        q0      =   x0[:nq]
        # Update robot model with initial state
        robot.framesForwardKinematics(q0)
        robot.computeJointJacobians(q0)

        ddp     =   init_ddp(robot=robot,config=config,x0=x0,actorResidualCritic=actorResidualCritic,N_h=horizon)



        if not use_warmstart:
            ug      = get_u_grav(q0, robot)
            xs_init = [x0 for i in range(horizon+1)]
            us_init = [ug  for i in range(horizon)]
            #xs_init = None
            #us_init = None
        else:
            xs_init, us_init = actorResidualCritic.warmstart(x0)


        ddp.solve(xs_init,us_init,maxiter=1000,isFeasible=False)

        is_outlier  =   outlier_check(ddp=ddp)



        if not is_outlier:

            init_x0s.append( x0 )
            v.append( ddp.cost )
            vx.append( np.array(ddp.Vx)[0] )
            xs.append(np.array(ddp.xs)[1:].flatten())
            us.append(np.array(ddp.us).flatten())
            iters_taken.append(ddp.iter)
            stopping_criteria.append(ddp.stoppingCriteria())

        desc = f"Datagen: ddp.cost:{np.round(ddp.cost,2)} ddp.iter:{ddp.iter} outlier:{is_outlier}"
        t_bar.set_description(desc)
        t_bar.refresh



    ddp = None
    
    datas = list(map(np.array, [init_x0s,v,vx,xs,us, iters_taken, stopping_criteria]))

    datas[1] = datas[1].reshape(-1,1,)

    return datas






