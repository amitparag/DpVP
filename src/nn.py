# Author: Amit Parag
# Date : 11 May, 2022


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import hessian as Hessian
import crocoddyl



class ActorResidualCritic(nn.Module):
    def __init__(self, x0s_dims:int=14, us_dims:int=7, max_action:float=20., horizon:int=5):
        super(ActorResidualCritic, self).__init__()

        """
        
                -->  | | | --> R --> R**2==V
        x0s     -->  | | | --> XS 
                -->  | | | --> US
        """

        self.x0s_dims = x0s_dims
        self.us_dims = us_dims
        self.horizon = horizon
        self.max_action = max_action

        ### Three networks
        # Value Function Residual Network
        self.residual_layers = nn.Sequential(
                nn.Linear(self.x0s_dims, 64),
                nn.ELU(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 3)
            )

        # State Policy Network
        self.state_layers = nn.Sequential(
            nn.Linear(self.x0s_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256,self.x0s_dims*self.horizon)
        )

        # Control Policy Network
        self.control_layers = nn.Sequential(
            nn.Linear(self.x0s_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.us_dims*self.horizon)
        )
         


    def forward(self, x, state=False,control=False,value=False):

        """Main forward function. OutPut depends on the kind of output wanted."""

        if value and not state and not control:

            r   =   self.residual_layers(x)
            v   = torch.sum(torch.square(r), axis=1).unsqueeze(1)
            vx  = torch.autograd.grad(outputs=v, inputs=x,
                                    retain_graph=True, grad_outputs=torch.ones_like(v),
                                    create_graph=True)[0]

            return v, vx

        elif state and not control and not value:
            return self.state_layers(x)

        elif control and not state and not value:
            return self.max_action*self.control_layers(x)

        elif value and state and control:

            r   =   self.residual_layers(x)
            v   =   torch.sum(torch.square(r), axis=1).unsqueeze(1)
            vx  =   torch.autograd.grad(outputs=v, inputs=x,
                                        retain_graph=True, grad_outputs=torch.ones_like(v),
                                        create_graph=True)[0]


            xs = self.state_layers(x)
            us = self.max_action*self.control_layers(x)

            return v, vx, xs, us

        elif state and control and not value:
            xs = self.state_layers(x)
            us = self.max_action*self.control_layers(x)

            return xs, us

        else:
            print("NO !!!")
            return -1



    ########## NUMPY FOR CROCODDYL


    def np2ten(self,x):
        """
        To make sure that x is a tensor
        """
        if not torch.is_tensor(x):
            x   =   torch.tensor(x,requires_grad=True,dtype=torch.float64).reshape(-1,self.x0s_dims)
        if x.ndim == 1:
            x = x.view(-1,self.x0s_dims)

        try:
            assert x.requires_grad == True
        except AssertionError:
            x = x.clone().detach().requires_grad_(True)
        return x



    def value_function(self,x,order=0):
        """
        Not to be used inside forward. For crocoddyl autograd hessian calculation
        """



        r = self.residual_layers(x)
                
        v = torch.sum(torch.square(r), axis=1).unsqueeze(1)
        
        if order == 1:
            dv = torch.autograd.grad(outputs = v,
                                    inputs = x, 
                                    retain_graph=True,grad_outputs=torch.ones_like(v), 
                                    create_graph=True)[0]

            return v, dv
        else:
            return v





    def terminal_value(self,x):
        x       =   self.np2ten(x)
        

        v,dv    =   self.value_function(x,order=1)
        d2v     =   Hessian(self.value_function, x.view(-1,self.x0s_dims),create_graph=True, \
                              strict=True).squeeze()

        dv          =   dv.detach().numpy().reshape(1,self.x0s_dims)
        d2v         =   d2v.detach().numpy().reshape(self.x0s_dims,self.x0s_dims)
        self.L      =   v.item()
        self.Lx     =   dv
        self.Lxx    =   d2v

    def warmstart(self,x):
        
        x0          =   self.np2ten(x)
        [xs,us]     =   list(self.forward(x=x0,state=True,control=True))
        
        
        xs      =    xs.detach().numpy().reshape(self.horizon,self.x0s_dims) 
        us      =    us.detach().numpy().reshape(self.horizon,self.us_dims) 
        
        
        x       =   [x] ## Initial state
        
            
        ### Warmstarted Solution
        x  +=   [np.array(xs_,dtype=np.float64) for xs_ in xs]
        
        us  =   [np.array(u,dtype=np.float64) for u in us]

        
        return x, us



class ActionModelTerminal(crocoddyl.ActionModelAbstract):
    """
    Terminal Model with a neural network that predicts cost, gradient and hessian at the terminal position.

    """
    def __init__(self,actorResidualCritic,nx:int=12,nu:int=6):
        """
        :param  agent                   torch nn, the value network to use.
        :param  nx                      int, number of input dimensions. dimensions of state space
        :param  nu                      int, dimensions of action space. This is only used to instantiate the
                                        C.ActionModelAbstract.
                                        The params nx and nu are needed by Crocoddyl to make a terminal model



        """
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(nx), nu)

        self.actorResidualCritic                  =   actorResidualCritic
       
    def calc(self,data,x,u=None):
        """Get the value function of the terminal position reached"""
        self.actorResidualCritic.terminal_value(x)

        data.cost       =  self.actorResidualCritic.L

    def calcDiff(self,data,x,u=None):
        """Get the gradient and the hessian of the value function"""
        data.Lx          =   self.actorResidualCritic.Lx.squeeze()
        data.Lxx         =   self.actorResidualCritic.Lxx.squeeze()
        




if __name__=='__main__':

    net = ActorResidualCritic(horizon=50)
    print(net)
    x = torch.rand(size=(100,14),requires_grad=True)
    xs,us = net(x,state=True,control=True,value=False)
    
    #xs, us = net.warmstart(x[1])
    print(xs.shape,'\t\t\n\n', us.shape)
    us = net(x,state=False,control=True,value=False)
    print(us.shape)












