import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters
# True probability of heads
htrue = 0.8
# Number of Trials
num_trials = 250

# Generate probablity possibilities (higher the thrid argument higher the approximation accuracy)
all_probabilities = np.linspace(0, 1, 2000)

# Dictionary storing priors
priors = {
    "non-informative": np.ones_like(all_probabilities) / len(all_probabilities),  
    "fair-coin": beta.pdf(all_probabilities, 5, 5),
    "weak-biased-heads" : 0.45 * beta.pdf(all_probabilities, 16, 4) + 0.55 * beta.pdf(all_probabilities, 12, 8),
    "strong-biased-tails" : beta.pdf(all_probabilities, 10, 190),
    "hyper-strong-biased-tails" : np.where(all_probabilities < 0.5, 2, 0)
}

# Likelihood Computation
def likelihood(num_heads, total_tosses, all_probabilities):
    #Computes the likelihood for observing num_heads given the probabilities.
    return (all_probabilities**num_heads) * ((1-all_probabilities)**(total_tosses-num_heads))

# This function finds posterior and normalizes it
def find_posterior(prior, likelihood):
    # Finds posterior from likelihood and prior
    posterior = prior * likelihood
    normalization_factor = np.sum(posterior)
    posterior = posterior / normalization_factor
    return posterior

# Coin Toss Function
def coin_toss(prob_head=0.8, n=10):
    #Simulate n coin tosses
    random_numbers = np.random.uniform(0, 1, n)
    num_heads = np.sum(random_numbers < prob_head)
    return num_heads


for key, prior in priors.items():
    tot_wins = 0
    win_probabilities = []
    posterior_tracker = []  
    
    for i in range(num_trials):
        # Simulate
        num_heads = coin_toss(prob_head=htrue)
        # Find likelihood this iteration
        likeIt = likelihood(num_heads, 10, all_probabilities)

        # Find and update posterior list
        posterior = find_posterior(prior, likeIt)
        posterior_tracker.append(posterior)
        
        # Check winning condition (num_heads <=6)
        if num_heads <= 6:
            tot_wins += 1
        # Append the win probability to the current probabilities
        win_probabilities.append(tot_wins / (i + 1))

        # Set the current posterior as the new prior
        prior = posterior

    # Posterior Evolution Visualization
    plt.figure(figsize=(10, 5))
    for i in range(0, num_trials, num_trials // 10):
        plt.plot(all_probabilities, posterior_tracker[i], label=f'Trial {i+1}')
    plt.title(f'Posteriors for {key} Prior')
    plt.xlabel('Probability of Heads')
    plt.ylabel('Posterior P Distribution')
    plt.legend()
    plt.show()

    # Winning Probability Over Trials Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(win_probabilities)
    plt.title(f'Winning Probability Over Time for {key} Prior')
    plt.xlabel('Trial Number')
    plt.ylabel('Win Chance')
    plt.show()