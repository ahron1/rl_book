### this program implements a single run of the k-bandits problem. 

import numpy as np
import matplotlib.pyplot as plt

bandit_reward_dist_mean = 0
bandit_reward_dist_sigma = 1
n_bandits = 10
bandit_means = np.random.normal(bandit_reward_dist_mean, bandit_reward_dist_sigma, n_bandits )
bandit_sigma = 1
samples_per_bandit = 1000
epsilon = 0.1

def select_action():
    r = np.random.randn()
    if r < epsilon:
        action = np.random.randint(0,n_bandits)
    else:
        action = np.argmax(q_values)

    return action
    
def update_action_count(A_t):
    n_action[A_t] += 1

def update_action_reward_total(A_t, R_t):
    action_rewards[A_t] += R_t

def update_q():
    for i in range(0, n_bandits):
        if n_action[i] > 0:
            q_values[i] = round(action_rewards[i] / n_action[i], 3)

### Each action corresponds to one bandit (slot machine). The goal is to figure out which bandit leads to the highest actions. Over a few steps, the agent figures out the mean reward from each bandit and chooses the action with the highest reward.

q_values = np.zeros(n_bandits) ## Q-value corresponding to each action. 

action_rewards = np.zeros(n_bandits) ## total reward from Action a
n_action = np.zeros(n_bandits) ## number of times action a has been chosen. This is used to compute the mean reward (q-value) from action a

n_steps = 1000
rewards_episodes = np.zeros(n_steps)
for i in range(0, n_steps):
    A_t = select_action()
    R_t = np.random.normal(bandit_means[A_t], bandit_sigma)
    R_t = round(R_t, 3)
    rewards_episodes[i] = R_t

    update_action_reward_total(A_t, R_t)
    update_action_count(A_t)
    update_q()



#    print("Action: ", A_t)
#    print("Reward: ", R_t)
#    print("Actions count - ", n_action)
#    print("Reward per action - ", action_rewards)
#    print("Q table - ", q_values)
#    print("-------------")

print("Q table - ", q_values)
print("Actions count - ", n_action)
print("Reward per action - ", action_rewards)


plt.plot(bandit_means)
plt.plot(q_values)
plt.show()

plt.plot(rewards_episodes)
plt.show()
