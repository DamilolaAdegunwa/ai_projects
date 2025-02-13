import numpy as np

# Example dataset: Age, Height (cm), Weight (kg), Income (USD)
sample_data = np.array([
    [25, 180, 75, 50000],
    [32, 165, 68, 48000],
    [40, 170, 72, 55000],
    [23, 175, 78, 52000]
])

# Separate target variable (Income)
Features = sample_data[:, :3]  # Features: Age, Height, Weight
Target = sample_data[:, 3]  # Target: Income


# Function to calculate correlation matrix.
# (addition) the data was first z-score normalized (meaning (data - mean)/ std), then the transpose of the data was then ran thru np.corrcoef(data)
def compute_correlation_matrix(data):
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    standardized_data = (data - mean_vals) / std_vals
    t_corrcoef = np.corrcoef(standardized_data.T)
    return t_corrcoef


# Function to normalize data (min-max scaling)
def normalize_data(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    min_max = (data - min_vals) / (max_vals - min_vals)
    return min_max


# Function to compute covariance matrix
def compute_covariance_matrix(data):
    mean_vals = np.mean(data, axis=0)
    centered_data = data - mean_vals  # this is called centering because it shifts the centered data mean to 0
    return np.cov(centered_data, rowvar=False)


# Linear regression using Numpy
def linear_regression(X, y):
    X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    return coeffs


def output():
    data = sample_data
    # Separate target variable (Income)
    X = Features
    y = Target

    # The raw data
    print("\ndata (Features: Age, Height, Weight - Target: Income):")
    print(data)

    # Compute correlation matrix
    correlation_matrix = compute_correlation_matrix(data)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Normalize data
    normalized_data = normalize_data(data)
    print("\nNormalized Data:")
    print(normalized_data)

    # Compute covariance matrix
    covariance_matrix = compute_covariance_matrix(data)
    print("\nCovariance Matrix:")
    print(covariance_matrix)

    # Perform linear regression
    regression_coeffs = linear_regression(X, y)
    print("\nLinear Regression Coefficients:")
    print(regression_coeffs)


# Test the functions with example inputs
if __name__ == "__main__":
    output()
