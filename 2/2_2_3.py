### this program implements n-runs of the k-bandit problem, for 3 different epsilon values

import numpy as np
import matplotlib.pyplot as plt

bandit_reward_dist_mean = 0
bandit_reward_dist_sigma = 1
k_bandits = 10
bandit_sigma = 1
samples_per_bandit = 1000
epsilons = [0, 0.01, 0.1]
n_epsilons = len(epsilons)
n_steps = 2000 #number of steps in each episode
n_trials = 2000 #each trial run n_steps with a fresh batch of bandits

def select_action(epsilon):
    r = np.random.rand()
    if r < epsilon:
        action = np.random.randint(0,k_bandits)
    else:
        action = np.argmax(q_estimates)

    return action
    
def update_action_count(A_t):
    # number of times each action has been taken so far
    n_action[A_t] += 1

def update_action_reward_total(A_t, R_t):
    # total reward from each action so far
    action_rewards[A_t] += R_t

def generate_reward(mean, sigma):
    # draw the reward from the normal distribution for this specific bandit 
    r = np.random.normal(mean, sigma)
    return r

def update_q(A_t, R_t):
#   method1 - use alpha
#   q_estimates[A_t] += 0.1 * (R_t - q_estimates[A_t])

#   method2 - use the equation in the book
    q_estimates[A_t] = action_rewards[A_t] / n_action[A_t]

average_reward_per_eplison = np.zeros((n_epsilons, n_steps))

for e in range(0, n_epsilons):
    rewards_episodes_trials = np.zeros((n_trials, n_steps))
    for j in range(0, n_trials):

        # pick the e-th epsilon
        epsilon = epsilons[e]

        # new bandits testbed
        q_true = np.random.normal(bandit_reward_dist_mean, bandit_reward_dist_sigma, k_bandits )

        # Q-value of each action (bandit) - start with random
        q_estimates = np.random.randn(k_bandits)

        ### Each action corresponds to one bandit (slot machine). The goal is to figure out which bandit leads to the highest actions. Over a few steps, the agent figures out the mean reward from each bandit and chooses the action with the highest reward.

        # Total reward from each action (bandit) - start with zeros
        action_rewards = np.zeros(k_bandits) 

        # number of times each action has been taken so far - start with zeros. This is used to compute the mean reward (q-value) from action a
        n_action = np.zeros(k_bandits) 

        # reward from each step - start from 0 
        rewards_episodes = np.zeros(n_steps)

        for i in range(0, n_steps):
            A_t = select_action(epsilon)
            R_t = generate_reward(q_true[A_t], bandit_sigma)
            rewards_episodes[i] = R_t

            update_action_reward_total(A_t, R_t)
            update_action_count(A_t)
            update_q(A_t, R_t)

        rewards_episodes_trials[j,:] = rewards_episodes

    average_reward_per_step = np.zeros(n_steps)
    for i in range(0, n_steps):
        average_reward_per_step[i] = np.mean(rewards_episodes_trials[:,i])

    average_reward_per_eplison[e, :] = average_reward_per_step


for i in range(0, n_epsilons):
    label = "epsilon : " + str(epsilons[i])
    plt.plot(average_reward_per_eplison[i], label = label)
plt.legend()
plt.show()
