# A set of utilities for loading robot
# Author : Amit Parag
# Date : 09/03/2022 Toulouse

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from optimal_control.path import load_config_file
from optimal_control.path import kuka_mesh_path
from optimal_control.path import kuka_urdf_path




def load_robot(display=False):
	"""
	Loads and returns the complete or the reduced Kuka Arm
	
	"""
	##################### Load the complete robot before solving OCP
	robot 					=	RobotWrapper.BuildFromURDF(kuka_urdf_path(), kuka_mesh_path())
	config,reduced_config 	=	load_config_file('static_reaching_task_ocp2')
	q0 						=	np.asarray(config['q0'])


	if config['reduce_it']:
		robot 	=	robot.buildReducedRobot(list_of_joints_to_lock=[config['idx_lock']],
	                                        reference_configuration=q0)

		config 	=	reduced_config # Reduced Config
		q0 		=	np.asarray(config['q0'])


	v0 			=	np.asarray(config['dq0'])
	x0 			=	np.concatenate([q0, v0])   
	id_endeff 	=	robot.model.getFrameId('contact')
	nq, nv 		=	robot.model.nq, robot.model.nv
	nx 			=	nq+nv
	nu          =	nq


	
	### INIT DISPLAY
	if display:
		robot.initViewer(loadModel=True)
		robot.framesForwardKinematics(q0)
		robot.computeJointJacobians(q0)
		robot.display(q0)
		viewer 	=	robot.viz.viewer
		gui		= 	viewer.gui
		#gui.createSceneWithFloor("world")
		
		M_des 				=	robot.data.oMf[id_endeff].copy()
		M_des.translation 	=	np.array(config['p_des'])
		tf_des				=	pin.utils.se3ToXYZQUAT(M_des)



		EPS_P = [0.05, 0.15, 0.25]
		EPS_V = [0.005, 0.01, 0.015]

		if (not gui.nodeExists("world/p_des")):
			gui.addSphere('world/p_des', .02, [1. ,0 ,0, 1.])

		gui.applyConfiguration('world/p_des',tf_des)


	print("\n\nSuccessfully Loaded Robot")
	return robot,config	

if __name__=='__main__':

	load_robot(display=True)