# A Supervised Approach to Reinforcement Learning
Reinforcement Learning sets it goal as the search for an optimal policy to navigate its immediate environment. To achieve optimal policy, it treats simulators as abstract oracles. In this project, we formulate a actor-critic esque solution of Markov Decision Process that combines a trajectory optimizer with a supervised learning phase, eliminating the need for extensive hyper parameter optimization, huge datasets and humongous training time.
The source code is released under the [MIT license](LICENSE).

The corresponding publication is available [here](https://hal.archives-ouvertes.fr/hal-03674092v2/document)

**Authors:** [Amit Parag](https://scholar.google.com/citations?user=wsRIfL4AAAAJ&hl=en&oi=ao) <br />
**Instructors:** [Nicolas Mansard](https://scholar.google.com/citations?user=rq-9xAkAAAAJ&hl=en) <br />

<br /> 

<img align="center" alt="" src="results/kuka.jpg" />  

<br /> 

The algorithm is implemented on the Kuka arm. The goal is to reach for the static target.
<br /> 
<br /> 
## Required Packages:
The Unsual Suspects
* [PyTorch](https://pytorch.org/)
* [Crocoddyl](https://github.com/loco-3d/crocoddyl)
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
* [example-robot-data](https://github.com/gepetto/example-robot-data)
* [Gepetto Gui](https://github.com/Gepetto/gepetto-viewer-corba) (optional for animating the learned policy in GUI)

The Usual Suspects
* [Numpy](https://numpy.org/)
* [tqdm](https://github.com/tqdm/tqdm)
* [Seaborn](https://seaborn.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [PyYaml](https://pypi.org/project/PyYAML/)
* [jupyter](https://jupyter.org/) (optional for notebooks)
<br /> 

## How to run the experiment

Clone the repo
```
git clone https://gitlab.laas.fr/aparag/kuka-arm-dpvp
```



Change directory to src

```
cd dpvp/src

```

Look at the exp.yml file for experiment params.

 Then run the main.py file
```
python3 main.py
```

The trained NN will be saved in `results/exp_`

The directory `config/robot_properties_kuka` contains URDF and meshes information of the robot, and `config/ocp_params` contains sets of robot parameters describing the OCP for Crocoddyl. The OCP itself is setup in `utils/ddp.py`

<br>

## Acknowledgements
You can find additional information on a part of this algorithm in [this paper](https://hal.archives-ouvertes.fr/hal-03356261/document) :

Parag, A., Kleff, S., Saci, L., Mansard, N., & Stasse, O. Value learning from trajectory optimization and Sobolev descent : A step toward reinforcement learning with superlinear convergence properties, _International Conference on Robotics and Automation (ICRA) 2022



