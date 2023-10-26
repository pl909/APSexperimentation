import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm

# Set parameters
theta_true = 3
K = 100  # number of measurements
nyquist_interval = 1/(24)  
t = np.arange(0, K * nyquist_interval, nyquist_interval)  # time vector
# Range of all thetas
thetas = np.linspace(0, 5, 500)


# Simulate 1 sample.
def get_sample(theta, variance):
    noise = np.random.normal(0, np.sqrt(variance), len(t))
    x = np.sin(theta * t) + noise
    return x
# Take prior, data, and compute posterior
def get_posterior(samp, prior, variance):
    likelihood = np.exp(-0.5 * np.sum((samp[:,None] - np.sin(thetas * t[:, None]))**2 / variance, axis=0))
    posterior = likelihood * prior(thetas)
    return posterior / np.trapz(posterior, thetas)  # Normalize the posterior
# Run three simulations for each scenario

scenarios = [
    ("Uniform Prior and Var = 2",2, lambda theta: uniform.pdf(theta, 0, 5)),
    ("Uniform Prior and Var = 0.5", 0.5, lambda theta: uniform.pdf(theta, 0, 5)),
    ("Gaussian(2,0.2) and Var is 2",2, lambda theta: norm.pdf(theta, 2, np.sqrt(0.2))),
    ("Gaussian(2,0.2) and Var is 0.5",0.5, lambda theta: norm.pdf(theta, 2, np.sqrt(0.2)))
]

for part, varConstant, prior in scenarios:
    plt.figure(figsize=(10, 8))
    for i in range(3):
        samples = get_sample(theta_true, varConstant)
        posterior = get_posterior(samples, prior, varConstant)

        plt.subplot(3, 2, i*2 + 1)
        plt.plot(t, samples)
        plt.title(f'{part}, Realization {i+1}: Data')
        plt.xlabel('Time')
        plt.ylabel('Measurement')

        plt.subplot(3, 2, i*2 + 2)
        plt.plot(thetas, posterior)
        plt.title(f'{part}, Realization {i+1}: Posterior')
        plt.xlabel('Theta')
        plt.ylabel('Posterior')
    
    plt.tight_layout()
    plt.show()
