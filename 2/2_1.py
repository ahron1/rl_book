import numpy as np
import matplotlib.pyplot as plt

bandit_reward_dist_mean = 0
bandit_reward_dist_sigma = 1
n_bandits = 10
bandit_means = np.random.normal(bandit_reward_dist_mean, bandit_reward_dist_sigma, n_bandits )
bandit_sigma = 1
samples_per_bandit = 1000

data = [np.random.normal(mean, bandit_sigma, size=samples_per_bandit ) for mean in bandit_means]
plt.violinplot(data, points = samples_per_bandit, widths = 0.5 )
plt.show()

