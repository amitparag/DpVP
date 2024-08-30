# This file contains utilities for loading the kuka robot from the configs folder.
# Author : Amit Parag
# Date : 09/03/2022 Toulouse

import yaml
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Load a yaml file (e.g. simu config file)
def load_yaml_file(yaml_file):
    '''
    Load config file (yaml)
    '''
    with open(yaml_file) as f:
        data = yaml.load(f)
    return data 

# Load config file
def load_config_file(config_name):
    '''
    Loads YAML config file in demos/config as a dict
    '''

    path    =   Path(__file__).resolve().parent.parent.parent

    
    config_path     =   os.path.abspath(os.path.join(path, 'configs/ocp_params'))
    config_file     =   config_path+"/"+config_name+".yml"
    config          =   load_yaml_file(config_file)


    if config['reduce_it']:
        reduced_config_file     =   config_path+"/"+config_name+"_reduced.yml"
        reduced_config          =   load_yaml_file(reduced_config_file)
    
        return config, reduced_config

    else:
        return config,0




# Get robot properties paths
def kuka_urdf_path():
    path = Path(__file__).resolve().parent.parent.parent

    return os.path.abspath(os.path.join(path, 'configs/robot_properties_kuka/iiwa.urdf'))

def kuka_mesh_path():
    path = Path(__file__).resolve().parent.parent.parent

    return os.path.abspath(os.path.join(path, 'configs/robot_properties_kuka'))


def exp_path(exp_number,make_path=True,get_path=True):

    path     =   Path(__file__).resolve().parent.parent.parent
    path     =   os.path.abspath(os.path.join(path,f'results/exp_{exp_number}/'))

    if make_path:
        print("Creating exp dir .. \n")
        p   =   Path(path)
        p.mkdir(parents=True,exist_ok=True)
    

    if get_path:
        return str(path)

if __name__=='__main__':
    exp_path(1,True)




