import math
import random

def gaussian_data_generator(mean, variance):
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean + math.sqrt(variance) * z0

mean_input = float(input("Enter the expectation value or mean: "))
variance_input = float(input("Enter the variance: "))

# Initialize estimates and data count
mean_estimate = 0
variance_estimate = 0
n = 0

# Threshold for convergence
threshold = 0.05

while True:
    n += 1
    x = gaussian_data_generator(mean_input, variance_input)
    
    # Sequentially update the mean and variance estimates
    delta = x - mean_estimate
    mean_estimate += delta / n
    variance_estimate += delta * (x - mean_estimate) #Welford's method

    # Printing the added point and the updated estimates
    print(f"Added Point {n}: {x}")
    print(f"Updated Mean: {mean_estimate}")
    print(f"Updated Variance: {variance_estimate / n}\n")  # Adjusted variance estimate

    if abs(mean_estimate - mean_input) <= threshold and abs(variance_estimate / n - variance_input) <= threshold:
        break

print(f"Converged Sequentially Estimated Mean: {mean_estimate}")
print(f"Converged Sequentially Estimated Variance: {variance_estimate / n}")  # Adjusted variance estimate


