# A set of utilities for viewing the solution in gepetto-gui.
# Author : Amit Parag
# Date : 09/03/2022 Toulouse

import numpy as np
import time
import pinocchio as pin


def animate_ddps(DDPS_DATA,robot, config):
    """Gepetto-gui must be open and running in terminal"""

    if ( type(DDPS_DATA) != list): DDPS_DATA = [DDPS_DATA]
    
    id_ee 			= 	robot.model.getFrameId('contact')
    robot.initViewer(loadModel=True)
    robot.display(np.asarray(config['q0']))

    viewer 	= robot.viz.viewer
    gui 	= viewer.gui
    
    
    for ddp_data in DDPS_DATA:

        M_des = robot.data.oMf[id_ee].copy()
        M_des.translation = np.asarray(config['p_des'])
        tf_des = pin.utils.se3ToXYZQUAT(M_des)
        #gui.createSceneWithFloor("world")
        if(not gui.nodeExists('world/p_des')):
            gui.addSphere('world/p_des', .02, [1. ,0 ,0, 1.])  
        gui.applyConfiguration('world/p_des', tf_des)
        q = np.array(ddp_data['xs'])[:,:ddp_data['nq']]
        for i in range(ddp_data['T']+1):
            robot.display(q[i])
            gui.refresh()
            time.sleep(0.05)