#这个是MCMC的简单实现，是copilot实现的


import numpy as np
import matplotlib.pyplot as plt

# 目标分布：标准正态分布
def target_dist(x):
    return np.exp(-0.5 * x ** 2)

# MCMC参数
n_samples = 10000
samples = np.zeros(n_samples)
current = 0.0

for i in range(1, n_samples):
    proposal = current + np.random.normal(0, 1)
    acceptance = min(1, target_dist(proposal) / target_dist(current))
    if np.random.rand() < acceptance:
        current = proposal
    samples[i] = current

# 可视化结果
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(samples, lw=0.5)
plt.title('MCMC Chain')
plt.xlabel('Step')
plt.ylabel('Value')

plt.subplot(1, 2, 2)
plt.hist(samples, bins=50, density=True, alpha=0.7, label='MCMC Samples')
x = np.linspace(-4, 4, 100)
plt.plot(x, 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2), 'r', label='True PDF')
plt.title('Histogram')
plt.legend()
plt.tight_layout()
plt.show()
