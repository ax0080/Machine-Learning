import numpy as np
import matplotlib.pyplot as plt

def polynomial_basis_linear_model_data_generator(n, coefficients, a):
    x = 2 * np.random.rand() - 1
    y = sum(coefficients[i] * (x ** i) for i in range(n))
    y += np.random.normal(0, a)
    return x, y

def bayesian_update(mu_prior, sigma_prior, x, y, a):
    phi_x = np.array([x**i for i in range(len(mu_prior))])
    sigma_posterior_inv = np.linalg.inv(sigma_prior) + (1/a**2) * np.outer(phi_x, phi_x)
    sigma_posterior = np.linalg.inv(sigma_posterior_inv)
    mu_posterior = sigma_posterior @ (np.linalg.inv(sigma_prior) @ mu_prior + (1/a**2) * y * phi_x)
    return mu_posterior, sigma_posterior

def check_convergence(mu, true_coefficients, threshold=0.1):
    """Check if the posterior mean has converged to the true coefficients."""
    differences = np.abs(mu - np.array(true_coefficients))
    return np.all(differences <= threshold)

def plot_function(x_vals, true_coefficients, mu, sigma, n, title, added_points=None):
    plt.figure(figsize=(8, 6))
    
    y_true = np.array([sum(true_coefficients[i] * (x ** i) for i in range(n)) for x in x_vals])
    y_mean = np.array([sum(mu[i] * (x ** i) for i in range(n)) for x in x_vals])
    y_variance = np.array([np.outer(np.array([x**i for i in range(n)]), np.array([x**i for i in range(n)])).flatten() @ sigma.flatten() for x in x_vals])
    
    if "Ground Truth" in title:
        plt.plot(x_vals, y_true, 'g-', label="Ground Truth")
        plt.fill_between(x_vals, y_true - y_variance, y_true + y_variance, color='red', alpha=0.2)
    else:
        plt.plot(x_vals, y_mean, 'k-', label="Mean of Function")
        plt.plot(x_vals, y_mean + y_variance, 'r-', label="Mean + 1 Variance")
        plt.plot(x_vals, y_mean - y_variance, 'r-', label="Mean - 1 Variance")
        if added_points is not None:
            plt.scatter(added_points[0], added_points[1], c='blue', marker='o', s=50, label="Added Points")

    plt.xlim(-2, 2)  # 限制x範圍在[-2, 2]
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    b = float(input("請輸入prior precision b: "))
    n = int(input("請輸入多項式的階數 n: "))
    a = float(input("請輸入高斯雜訊的標準差 a: "))
    
    true_coefficients = [float(input(f"請輸入真實的W{i} :")) for i in range(n)]
    
    mu_prior = np.zeros(n)
    sigma_prior = np.eye(n) * (1/b)
    
    x_vals = np.linspace(-2, 2, 400)  # 設定x_vals範圍為[-2, 2]
    plot_function(x_vals, true_coefficients, mu_prior, sigma_prior, n, "Ground Truth with Initial Variance")
    
    added_xs = []
    added_ys = []

    while not check_convergence(mu_prior, true_coefficients):
        x, y = polynomial_basis_linear_model_data_generator(n, true_coefficients, a)
        added_xs.append(x)
        added_ys.append(y)
        mu_prior, sigma_prior = bayesian_update(mu_prior, sigma_prior, x, y, a)

        if len(added_xs) == 10:
            plot_function(x_vals, true_coefficients, mu_prior, sigma_prior, n, "After 10 Incomes", (added_xs, added_ys))
        elif len(added_xs) == 50:
            plot_function(x_vals, true_coefficients, mu_prior, sigma_prior, n, "After 50 Incomes", (added_xs, added_ys))
        
    plot_function(x_vals, true_coefficients, mu_prior, sigma_prior, n, "Final Converged Prediction", (added_xs, added_ys))

if __name__ == "__main__":
    main()