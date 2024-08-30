# This script uses Crocoddyl to solve Optimal control problems 
# Author: Amit Parag
# Date : 11 May, 2022 Toulouse




import crocoddyl
import numpy as np
import pinocchio as pin
from tqdm import trange

from optimal_control.utils import get_p
from optimal_control.samples import uniform_samples
from optimal_control.robot import load_robot

from nn import ActionModelTerminal


def init_ddp(robot=None,config=None,x0=None,actorResidualCritic=None, dt=None, N_h=None):

    if robot is None:

        robot, config   =   load_robot()

    # OCP parameters
    if(dt is None):
        dt = config['dt']                   # OCP integration step (s)

    if(N_h is None):
        N_h = config['N_h']                 # Number of knots in the horizon

    # Model params
    id_endeff = robot.model.getFrameId('contact')
    M_ee = robot.data.oMf[id_endeff]
    nq, nv = robot.model.nq, robot.model.nv
    # Construct cost function terms
    # State and actuation models
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)

    which_costs = config['WHICH_COSTS']

    # State regularization
    if('all' in which_costs or 'stateReg' in which_costs):
        stateRegWeights = np.asarray(config['stateRegWeights'])
        x_reg_ref = np.concatenate(
            [np.asarray(config['q0']), np.asarray(config['dq0'])])  # np.zeros(nq+nv)
        xRegCost = crocoddyl.CostModelResidual(state,
                                               crocoddyl.ActivationModelWeightedQuad(
                                                   stateRegWeights**2),
                                               crocoddyl.ResidualModelState(state, x_reg_ref, actuation.nu))
    # Control regularization
    if('all' in which_costs or 'ctrlReg' in which_costs):
        ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
        uRegCost = crocoddyl.CostModelResidual(state,
                                               crocoddyl.ActivationModelWeightedQuad(
                                                   ctrlRegWeights**2),
                                               crocoddyl.ResidualModelControlGrav(state))
    # State limits penalization
    if('all' in which_costs or 'stateLim' in which_costs):
        x_lim_ref = np.zeros(nq+nv)
        q_max = 0.95*state.ub[:nq]  #  95% percent of max q
        v_max = np.ones(nv)  #  [-1,+1] for max v
        x_max = np.concatenate([q_max, v_max])  #  state.ub
        stateLimWeights = np.asarray(config['stateLimWeights'])
        xLimitCost = crocoddyl.CostModelResidual(state,
                                                 crocoddyl.ActivationModelWeightedQuadraticBarrier(
                                                     crocoddyl.ActivationBounds(-x_max, x_max), stateLimWeights),
                                                 crocoddyl.ResidualModelState(state, x_lim_ref, actuation.nu))
    # Control limits penalization
    if('all' in which_costs or 'ctrlLim' in which_costs):
        u_min = -np.asarray(config['ctrl_lim'])
        u_max = +np.asarray(config['ctrl_lim'])
        u_lim_ref = np.zeros(nq)
        uLimitCost = crocoddyl.CostModelResidual(state,
                                                 crocoddyl.ActivationModelQuadraticBarrier(
                                                     crocoddyl.ActivationBounds(u_min, u_max)),
                                                 crocoddyl.ResidualModelControl(state, u_lim_ref))
        # print("[OCP] Added ctrl lim cost.")
    # End-effector placement
    if('all' in which_costs or 'placement' in which_costs):
        p_target = np.asarray(config['p_des'])
        desiredFramePlacement = pin.SE3(M_ee.rotation, p_target)
        framePlacementWeights = np.asarray(config['framePlacementWeights'])
        framePlacementCost = crocoddyl.CostModelResidual(state,
                                                         crocoddyl.ActivationModelWeightedQuad(
                                                             framePlacementWeights**2),
                                                         crocoddyl.ResidualModelFramePlacement(state,
                                                                                               id_endeff,
                                                                                               desiredFramePlacement,
                                                                                               actuation.nu))
    # End-effector velocity
    if('all' in which_costs or 'velocity' in which_costs):
        desiredFrameMotion = pin.Motion(np.concatenate(
            [np.asarray(config['v_des']), np.zeros(3)]))
        frameVelocityWeights = np.ones(6)
        frameVelocityCost = crocoddyl.CostModelResidual(state,
                                                        crocoddyl.ActivationModelWeightedQuad(
                                                            frameVelocityWeights**2),
                                                        crocoddyl.ResidualModelFrameVelocity(state,
                                                                                             id_endeff,
                                                                                             desiredFrameMotion,
                                                                                             pin.LOCAL,
                                                                                             actuation.nu))
    # Frame translation cost
    if('all' in which_costs or 'translation' in which_costs):
        desiredFrameTranslation = np.asarray(config['p_des'])
        frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
        frameTranslationCost = crocoddyl.CostModelResidual(state,
                                                           crocoddyl.ActivationModelWeightedQuad(
                                                               frameTranslationWeights**2),
                                                           crocoddyl.ResidualModelFrameTranslation(state,
                                                                                                   id_endeff,
                                                                                                   desiredFrameTranslation,
                                                                                                   actuation.nu))

    # Create IAMs
    runningModels = []
    for i in range(N_h):
        # Create IAM
        runningModels.append(crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state,
                                                             actuation,
                                                             crocoddyl.CostModelSum(state, nu=actuation.nu)), dt))
        # Add cost models
        if('all' in which_costs or 'placement' in which_costs):
            runningModels[i].differential.costs.addCost(
                "placement", framePlacementCost, config['framePlacementWeight'])
        if('all' in which_costs or 'translation' in which_costs):
            runningModels[i].differential.costs.addCost(
                "translation", frameTranslationCost, config['frameTranslationWeight'])
        if('all' in which_costs or 'velocity' in which_costs):
            runningModels[i].differential.costs.addCost(
                "velocity", frameVelocityCost, config['frameVelocityWeight'])
        if('all' in which_costs or 'stateReg' in which_costs):
            runningModels[i].differential.costs.addCost(
                "stateReg", xRegCost, config['stateRegWeight'])
        if('all' in which_costs or 'ctrlReg' in which_costs):
            runningModels[i].differential.costs.addCost(
                "ctrlReg", uRegCost, config['ctrlRegWeight'])
        if('all' in which_costs or 'stateLim' in which_costs):
            runningModels[i].differential.costs.addCost(
                "stateLim", xLimitCost, config['stateLimWeight'])
        if('all' in which_costs or 'ctrlLim' in which_costs):
            runningModels[i].differential.costs.addCost(
                "ctrlLim", uLimitCost, config['ctrlLimWeight'])
        # Add armature
        runningModels[i].differential.armature = np.asarray(config['armature'])

    # Terminal IAM + set armature
    if actorResidualCritic is None:

        terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state,
                                                             actuation,
                                                             crocoddyl.CostModelSum(state, nu=actuation.nu)))

        terminalModel.differential.armature = np.asarray(config['armature'])

    else:
        terminalModel = ActionModelTerminal(actorResidualCritic=actorResidualCritic,nx=nq+nv,nu=nq)

    if x0 is None:
        nx = nq + nv
        x0 = np.zeros(nx,) + 1.5

        # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

    # Creating the DDP solver
    ddp = crocoddyl.SolverFDDP(problem)

    return ddp



def extract_ddp_data(ddp):
  """Take a ddp object and return a dictionary of ddp_data"""
  # Store data
  ddp_data = {}
  # OCP params
  ddp_data['T']   = ddp.problem.T
  ddp_data['dt']  = ddp.problem.runningModels[0].dt
  ddp_data['nq']  = ddp.problem.runningModels[0].state.nq
  ddp_data['nv']  = ddp.problem.runningModels[0].state.nv
  ddp_data['nu']  = ddp.problem.runningModels[0].differential.actuation.nu
  ddp_data['nx']  = ddp.problem.runningModels[0].state.nx
  # Pin model
  ddp_data['pin_model'] = ddp.problem.runningModels[0].differential.pinocchio
  ddp_data['frame_id']  = ddp.problem.runningModels[0].differential.costs.costs['translation'].cost.residual.id


  ## Extract value function along the trajectory
  cost = []
  for d in ddp.problem.runningDatas:
      cost.append(d.cost)
  cost.append(ddp.problem.terminalData.cost)
  for i, _ in enumerate(cost):
      cost[i] = sum(cost[i:])

  ddp_data["init_config"] = ddp.problem.x0
  ddp_data["values"] = cost
  ddp_data["value_function"] = ddp.cost
  ddp_data["iter"] = ddp.iter
  ddp_data["gradients"] = np.array(ddp.Vx)
  # Solution trajectories
  ddp_data['xs'] = np.array(ddp.xs)
  ddp_data['us'] = np.array(ddp.us)
  ddp_data['riccati_gains'] = np.array(ddp.K)
  ddp_data['sc'] = ddp.stoppingCriteria()
  
  ddp_data['q'] = ddp_data["xs"][:,:ddp_data["nq"]]
  ddp_data['v'] = ddp_data['xs'][:,ddp_data['nv']:] ########### Useful for diagostic test

  p_EE          = get_p(ddp_data['q'], ddp_data['pin_model'], ddp_data['frame_id'])
   
  ddp_data["p_EE"] = p_EE    
  ddp_data["p_EE_final"] = p_EE[-1]
  ddp_data["p_EE_initial"] = p_EE[0]



  return ddp_data
