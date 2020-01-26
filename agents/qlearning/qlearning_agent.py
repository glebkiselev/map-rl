import itertools
import sys
from collections import defaultdict
from ast import literal_eval

import numpy as np

from agents.agent import Agent


class QLearningAgent(Agent):
    """
    An implementation of the Q Learning agent.

    """

    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1, beta=0.2):
        self.environment = env
        self.number_of_action = env.action_space.n
        self.q = defaultdict(lambda: np.zeros(self.number_of_action))
        self.r_avg = 0
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.policy = self._make_epsilon_greedy_policy()

    def _make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        """

        def policy_fn(state):
            A = np.ones(self.number_of_action, dtype=float) * self.epsilon / self.number_of_action
            best_action = np.argmax(self.q[state])
            A[best_action] += (1.0 - self.epsilon)
            return A

        return policy_fn

    def act(self, state):
        action_probs = self.policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q[next_state])
        td_target = reward + self.gamma * self.q[next_state][best_next_action]
        td_delta = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_delta

    def rupdate(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q[next_state])
        td_target = reward - self.r_avg + self.q[next_state][best_next_action]
        td_delta = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_delta
        if action == np.argmax(self.q[state]):
            r_avg_delta = reward + self.q[next_state][best_next_action] - self.q[state][action]
            r_avg_delta -= self.r_avg
            self.r_avg += self.beta * r_avg_delta

    def train(self, num_episodes=500, verbose=False):
        total_total_reward = 0.0
        rewards = []
        for i_episode in range(num_episodes):

            # Print out which episode we're on.
            if (i_episode + 1) % 1 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            state = self.environment.reset()
            state = str(state)
            if i_episode == num_episodes-1:
                print('\rlol')

            # checking the speed of finding the optimal policy
            # grab = np.argmax(self.q['(5, 8, 0)'])
            # rotate = np.argmax(self.q['(5, 8, 1)'])
            # push = np.argmax(self.q['(5, 8, 2)'])
            # if grab == 4 and rotate == 5 and push == 6:
            #     print('optimal policy found')

            total_reward = 0.0
            for t in itertools.count():
                action = self.act(state)
                next_state, reward, done, _ = self.environment.step(action)
                next_state = str(next_state)
                # print("\rEpisode {}/{}, t={}.".format(i_episode + 1, num_episodes,t), end="")
                # print("\r{} {}".format(self.q['(6,0)'], t), end="")
                # sys.stdout.flush()
                total_reward += reward

                self.update(state, action, reward, next_state)

                if done:
                    total_total_reward += total_reward
                    rewards.append(total_reward)
                    break

                state = next_state
        return total_total_reward / num_episodes, rewards  # return average eps reward


class QLearningWithOptionsAgent(QLearningAgent):

    def __init__(self, env, options, gamma=1.0, alpha=0.5, epsilon=0.1, intra_options=False):
        super().__init__(env, gamma, alpha, epsilon)

        self.options = options
        self.number_of_action = env.action_space.n + len(options)
        self.q = defaultdict(lambda: np.zeros(self.number_of_action))

        self.intra_options = intra_options
        if intra_options:
            self.option_q_hat = defaultdict(lambda: np.zeros(self.number_of_action))

        self.policy = self._make_epsilon_greedy_policy()

    def train(self, num_episodes=500, verbose=False):
        total_total_reward = 0.0
        rewards = []
        for i_episode in range(num_episodes):

            # Print out which episode we're on.
            if (i_episode + 1) % 1 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            state = self.environment.reset()
            state = str(state)

            total_reward = 0.0
            for t in itertools.count():

                action = self.act(state)
                if action >= self.environment.action_space.n:
                    next_state, reward, done, _ = self._execute_option(state, action)
                else:
                    next_state, reward, done, _ = self.environment.step(action)
                    next_state = str(next_state)

                    if self.intra_options:

                        update_options = []
                        for i, o in enumerate(self.options):
                            o_a, _ = o.step(state)
                            if o_a == action:
                                update_options.append(i + self.environment.action_space.n)

                        # Intra option Q learning update
                        # self.q[state][action] = self.q[state][action] + self.alpha * (
                        #         (reward * self.gamma * (self.option_q_hat[next_state][action])) -
                        #         self.q[state][action])
                        #
                        # max_o = np.max(self.q[next_state])
                        # if done:
                        #     beta_s = 1
                        # else:
                        #     beta_s = 0
                        # self.option_q_hat[next_state][action] = (1 - beta_s) * self.q[next_state][
                        #     action] + beta_s * max_o

                        for o in update_options:
                            self.q[state][o] += self.alpha * (reward + self.gamma * (
                                    self.option_q_hat[next_state][o] - self.q[state][o]))
                            max_o = np.max(self.q[next_state])
                            if self.options[o - self.environment.action_space.n].termination_condition(
                                    state=next_state):
                                beta_s = 1
                            else:
                                beta_s = 0
                            self.option_q_hat[next_state][o] = (1 - beta_s) * self.q[next_state][
                                o] + beta_s * max_o

                total_reward += reward

                self.update(state, action, reward, next_state)

                if done:
                    total_total_reward += total_reward
                    rewards.append(total_reward)
                    break

                state = next_state
        return total_total_reward / num_episodes, rewards  # return average eps reward

    def _execute_option(self, state, action):
        done = False
        total_reward = 0
        total_steps = 0
        reward_for_false_execution = -2

        option = self.options[action - self.environment.action_space.n]
        if state not in option.initialisation_set:
            return state, reward_for_false_execution, done, None

        terminated = False

        while not terminated:
            a, terminated = option.step(state)
            next_state, reward, done, _ = self.environment.step(a)  # taking action
            next_state = str(next_state)

            total_reward += reward  # * (self.gamma ** total_steps)
            total_steps += 1

            if done:
                break

            state = next_state

        return state, total_reward, done, None
