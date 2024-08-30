# A Supervised Approach to Reinforcement Learning
Deep reinforcement learning uses simulators as abstract oracles to interact with the
environment. In continuous domains of multi body robotic systems, differentiable
simulators have recently been proposed but are yet under utilized, even though we
have the knowledge to make them produce richer information. This problem when
juxtaposed with the usually high computational cost of exploration-exploitation in
high dimensional state space can quickly render reinforcement learning algorithms
impractical. In this paper, we propose to combine learning and simulators such
that the quality of both increases, while the need to exhaustively search the state
space decreases. We propose to learn value function and state, control trajectories
through the locally optimal runs of model based trajectory optimizer. The learned
value function, along with an estimate of optimal state and control policies, is
subsequently used in the trajectory optimizer: the value function estimate serves as
a proxy for shortening the preview horizon, while the state and control approxima-
tions serve as a guide in policy search for our trajectory optimizer. The proposed
approach demonstrates a better symbiotic relation, with super linear convergence,
between learning and simulators, that we need for end-to-end learning of complex
poly articulated systems.

The source code is released under the [MIT license](LICENSE).

The corresponding publication is available [here](https://hal.archives-ouvertes.fr/hal-03674092v2/document)

**Authors:** [Amit Parag](https://scholar.google.com/citations?user=wsRIfL4AAAAJ&hl=en&oi=ao) <br />

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





