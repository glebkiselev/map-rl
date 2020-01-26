from ast import literal_eval
import time
import gym
import numpy as np
import envs.gridworld
import matplotlib.pyplot as plt

from utils.options import load_option

from agents.qlearning.qlearning_agent import QLearningAgent, QLearningWithOptionsAgent


def train(parameters, withOptions=False, intra_options=False):
    num_episodes = int(parameters['episodes'])
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']

    goal = "G1"
    if goal == "G2":
        env_name = "FourRooms-v2"
    else:
        env_name = "FourRooms-v1"
        goal = "G1"
    env_name = "FourRooms-v3"
    env = gym.make(env_name)

    if withOptions:
        # options = [load_option('FourRoomsO1'), load_option('FourRoomsO2')]
        options = [load_option('FourRoomsO3')]
        agent = QLearningWithOptionsAgent(env, options, gamma=gamma, alpha=alpha, epsilon=epsilon,
                                          intra_options=intra_options)
    else:
        agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    average_eps_reward, all_rewards = agent.train(num_episodes)

    # if withOptions and intra_options:
    #     options = agent.options
    #     for i, o in enumerate(options):
    #         env.render(draw_arrows=True, policy=o.policy,
    #                    name_prefix="Policy of Option " + str(i + 1) + "\n" + env_name + " (" + goal + ")")
    env.render(draw_arrows=True, policy=q_to_policy(agent.q),
               name_prefix=env_name)

    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.plot(all_rewards[-100:])
    plt.show()

    env.close()
    return average_eps_reward


def q_to_policy(q, offset=0):
    optimalPolicy = {}
    for state in q:
        optimalPolicy[state] = np.argmax(q[state]) + offset
    return optimalPolicy


def main():
    parameters = {'episodes': 1000, 'gamma': 0.9, 'alpha': 0.1, 'epsilon': 0.1}
    print('---Start---')
    start = time.time()
    average_reward = train(parameters, withOptions=True, intra_options=True)
    end = time.time()
    print('\nAverage reward: {}', average_reward)
    print('Time (', parameters['episodes'], 'episodes ):', end - start)
    print('---End---')


if __name__ == '__main__':
    main()
