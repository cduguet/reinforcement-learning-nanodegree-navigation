# Reinforcement Learning Project 1: Navigation

---

This projects implements a Reinforcement Learning Agent to Solve the Bananna Finding Problem, defined as the first chalenge of the [Udacity's Reinforcement Learning Nanodegree](udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

The model of the environment is built under Unity, and the Deep Learning Agent is written in Python, using PyTorch. 

## Installation

### Prerequisites

 - Python 3 and Pip
 - Install Anaconda or Miniconda 3 from [here](https://docs.conda.io/en/latest/miniconda.html)

### Installation
 - In the root folder folder, run `conda env create`. This will create a python environment to run the agent.
 - Activate the environment with `conda activate drlnd-navigation`.
 - Download and unzip the Unity environment executable in the folder from [here](https://classroom.udacity.com/nanodegrees/nd893/parts/6b0c03a7-6667-4fcf-a9ed-dd41a2f76485/modules/4eeb16ab-5ac5-47bf-974d-12784e9730d7/lessons/69bd42c6-b70e-4866-9764-9bfa8c03cdea/concepts/319dc918-bd2c-4d3b-80a5-063bb5f1905a)
   
   - For Linux (headless version, no visualization): 
   ```
   wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip & unzip Banana_Linux_NoVis.zip
   ```
   
   
   - For Linux (with visualization, [currently buggy](https://knowledge.udacity.com/questions/98593)): 
   ```
   wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip & unzip Banana_Linux.zip
   ```
## Usage

### Training
To train the agent from zero, run 
```
python train.py --env <path to environment> 
```
### Test
```
python test.py --env <path to environment> --checkpoint <path to checkpoint file>
```

## Troubleshooting 
If you have trouble getting the Visualization of the environment work, check that you are using a compatible version of the `mlagents` package, and that the python script is being called from am environment with the variale `DISPLAY` set to `:0`. For that you can prepend to the train script: 

```export DISPLAY=:0; python train.py --env <path to environment> 
```
## Report
A more detailed report of the inner workings of the agent and hyperparameters can be found [here](Report.ipynb)

