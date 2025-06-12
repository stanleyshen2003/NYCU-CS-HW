import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('CartPole-v0')
    plt.xlabel('epoch')
    plt.ylabel('rewards')


def taxi():
    plt.figure(figsize=(10, 5))
    plt.title('Taxi-v3')
    plt.xlabel('epoch')
    plt.ylabel('rewards')
    rewards = np.load("./Rewards/taxi_rewards.npy").transpose()
    rewards_avg = np.mean(rewards, axis=1)
    plt.plot([i for i in range(3000)], rewards_avg[:3000],
             label='taxi', color='gray')
    plt.legend(loc="best")
    plt.savefig("./Plots/taxi.png")
    plt.show()
    plt.close()


def cartpole():
    Q_learning_Rewards = np.load("./Rewards/cartpole_rewards.npy").transpose()
    Q_learning_avg = np.mean(Q_learning_Rewards, axis=1)
    Q_learning_std = np.std(Q_learning_Rewards, axis=1)
    initialize_plot()
    plt.plot([i for i in range(3000)], Q_learning_avg,
             label='cartpole', color='orange')
    plt.fill_between([i for i in range(3000)],
                     Q_learning_avg+Q_learning_std, Q_learning_avg-Q_learning_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/cartpole.png")
    plt.show()
    plt.close()


def DQN():
    DQN_Rewards = np.load("./Rewards/DQN_rewards.npy").transpose()
    DQN_avg = np.mean(DQN_Rewards, axis=1)
    DQN_std = np.std(DQN_Rewards, axis=1)

    initialize_plot()

    plt.plot([i for i in range(1000)], DQN_avg,
             label='DQN', color='blue')
    plt.fill_between([i for i in range(1000)],
                     DQN_avg+DQN_std, DQN_avg-DQN_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/DQN.png")
    plt.show()
    plt.close()


def compare():
    DQN_Rewards = np.load("./Rewards/DQN_rewards.npy").transpose()
    DQN_avg = np.mean(DQN_Rewards, axis=1)
    Q_learning_Rewards = np.load("./Rewards/cartpole_rewards.npy").transpose()
    Q_learning_avg = np.mean(Q_learning_Rewards, axis=1)
    initialize_plot()
    plt.plot([i for i in range(1000)], DQN_avg, label='DQN', color='blue')
    plt.plot([i for i in range(1000)],
             Q_learning_avg[:1000], label='Q_learning', color='orange')
    plt.legend(loc="best")
    plt.savefig("./Plots/compare.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    '''
    Plot the trend of Rewards
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxi", action="store_true")
    parser.add_argument("--cartpole", action="store_true")
    parser.add_argument("--DQN", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

        
    os.makedirs("./Plots", exist_ok=True)

    if args.taxi:
        taxi()
    elif args.cartpole:
        cartpole()
    elif args.DQN:
        DQN()
    elif args.compare:
        compare()
