### this program implements n-runs of the k-bandit problem

import numpy as np
import matplotlib.pyplot as plt

bandit_reward_dist_mean = 0
bandit_reward_dist_sigma = 1
k_bandits = 10
bandit_sigma = 1
samples_per_bandit = 1000
epsilon = 0.01

def select_action():
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
    #q_estimates[A_t] += 0.1 * (R_t - q_estimates[A_t])
    q_estimates[A_t] = action_rewards[A_t] / n_action[A_t]

n_steps = 1000
n_trials = 2000 #each trial run n_steps with a fresh batch of bandits

# matrix of rewards in each step across all the trials - start from zeros
rewards_episodes_trials = np.zeros((n_trials, n_steps))

for j in range(0, n_trials):
    # new bandits testbed
    q_true = np.random.normal(bandit_reward_dist_mean, bandit_reward_dist_sigma, k_bandits )
    #q_true = np.random.randn(k_bandits )
    # Q-value of each action (bandit) - start with random
    q_estimates = np.random.randn(k_bandits)
    # Total reward from each action (bandit) - start with zeros
    action_rewards = np.zeros(k_bandits) 
    # number of times each action has been taken so far - start with zeros
    n_action = np.zeros(k_bandits) 
    # reward from each step - start from 0 
    rewards_episodes = np.zeros(n_steps)
    for i in range(0, n_steps):
        A_t = select_action()
        R_t = generate_reward(q_true[A_t], bandit_sigma)
        rewards_episodes[i] = R_t

        update_action_reward_total(A_t, R_t)
        update_action_count(A_t)
        update_q(A_t, R_t)

    rewards_episodes_trials[j,:] = rewards_episodes

# average reward per step over all the runs
average_reward_per_step = np.zeros(n_steps)
for i in range(0, n_steps):
    average_reward_per_step[i] = np.mean(rewards_episodes_trials[:,i])


#    print("Action: ", A_t)
#    print("Reward: ", R_t)
#    print("Actions count - ", n_action)
#    print("Reward per action - ", action_rewards)
#    print("Q table - ", q_estimates)
#    print("-------------")

#print("Q table - ", q_estimates)
#print("Actions count - ", n_action)
#print("Reward per action - ", action_rewards)
#
#
#plt.plot(q_true)
#plt.plot(q_estimates)
#plt.show()

plt.plot(average_reward_per_step)
plt.show()
