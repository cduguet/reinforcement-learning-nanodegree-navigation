#!/usr/bin/env python
import argparse
from collections import deque
from datetime import date

import numpy as np
import torch
from unityagents import UnityEnvironment

from agent.agent import Agent

def dqn_trainer(agent, env, brain_name, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    eps = eps_start

    for i_episode in range(n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]

            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.1f}\tCurrent Score: {score:.2f}', end='')

        if i_episode % 100 == 0 and i_episode != 0:
            print('')
        if np.mean(scores[-100:]) >= 13.0:
            print(f'\nEnvironment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores[-100:]):.1f}')
            today = date.today()
            checkpoint_file = 'checkpoint' + today.isoformat() + '.pth'
            print(f'Saving results in {checkpoint_file}')
            agent.save_local_weights(checkpoint_file)
            break
    return scores            


def main(args):
    env = UnityEnvironment(file_name=args.env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)

    print('Action Space:', action_size)
    print('State Space: \n', state_size)

    agent = Agent(state_size=state_size, action_size=action_size)

    scores = dqn_trainer(agent, env, brain_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Deep Learning Agent for the Udacity Banana Environment")
    parser.add_argument( '--env', dest='env', help='Specify an executable environment file made with Unity', 
                        default='Banana_Linux/Banana.x86_64')
    args = parser.parse_args()
    main(args)
