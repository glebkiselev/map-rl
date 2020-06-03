import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_episode_stats(stats, title_prefix, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title(title_prefix + "\nEpisode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title(title_prefix + "\nEpisode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title(title_prefix + "\nEpisode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show()

    return fig1, fig2, fig3

def plot_average_reward(smoothing_window=10):
    import os
    import pickle
    plots = {}
    for file in os.listdir(os.getcwd()):
        if 'interval' in file:
            with open(file, 'rb') as f:
                rewards = pickle.load(f)
        if '30x30' in file:
            plots['30x30'] = rewards
        elif '40x40' in file:
            plots['40x40'] = rewards
        elif '50x50' in file:
            plots['50x50'] = rewards
    for name, rewards in plots.items():
        rewards_smoothed = pd.Series(rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(range(len(rewards_smoothed)), rewards_smoothed, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Rewards")
    plt.legend()
    plt.title("Q-learning rewards for maps of various sizes")
    plt.show()

plot_average_reward()