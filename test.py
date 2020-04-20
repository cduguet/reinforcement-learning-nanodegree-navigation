#!/usr/bin/env python
import argparse
from collections import deque
from datetime import date

import numpy as np
import torch
from unityagents import UnityEnvironment

from agent.agent import Agent

def dqn_tester(agent, env, brain_name):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    eps = 0
    while True:
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break
    print(f'Finishing score: {score}.')
    return score

def main(args):
    env = UnityEnvironment(file_name=args.env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = Agent(state_size=state_size, action_size=action_size)
    agent.load_local_weights(args.checkpoint)

    scores = dqn_tester(agent, env, brain_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Deep Learning Agent for the Udacity Banana Environment")
    parser.add_argument( '--env', dest='env', help='Specify an executable environment file made with Unity', 
                        default='Banana_Linux/Banana.x86_64')
    parser.add_argument( '--checkpoint', dest='checkpoint', help='Checkpoint where the agent has been saved', 
                        default='checkpoint.pth')
    
    args = parser.parse_args()
    main(args)
