import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) from scratch.

    Parameters:
    - y_true: array-like, shape (n_samples,) - True values.
    - y_pred: array-like, shape (n_samples,) - Predicted values.

    Returns:
    - mse: float - MSE.
    """
    squared_diff = (y_true - y_pred) ** 2
    mse = np.mean(squared_diff)
    return mse


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) from scratch.

    Parameters:
    - y_true: array-like, shape (n_samples,) - True values.
    - y_pred: array-like, shape (n_samples,) - Predicted values.

    Returns:
    - rmse: float - RMSE.
    """    
    squared_diff = (y_true - y_pred) ** 2
    rmse = np.sqrt(np.mean(squared_diff))
    return rmse


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) from scratch.

    Parameters:
    - y_true: array-like, shape (n_samples,) - True values.
    - y_pred: array-like, shape (n_samples,) - Predicted values.

    Returns:
    - mae: float - MAE.
    """
    absolute_diff = np.abs(y_true - y_pred)
    mae = np.mean(absolute_diff)
    return mae


def calculate_r2(y_true, y_pred):
    """
    Calculates the R^2 score from scratch.

    Parameters:
        y_true: array-like, shape (n_samples,) - True values.
        y_pred: array-like, shape (n_samples,) - Predicted values.

    Returns:
        r2: float - R^2 score.
    """
    mean_true = np.mean(y_true)
    
    # Sum of squared total variation
    TSS = np.sum((y_true - mean_true) ** 2)
    
    # Sum of squared residuals
    RSS = np.sum((y_true - y_pred) ** 2)
    
    # R^2 score calculation
    r2 = 1 - (RSS / TSS)
    
    return r2


def regression_metrics(y_true, y_pred, model_name=''):
    """
    Calculates scores using the common regression metrics for the given actual and predicted values.
    
    Parameters:
        y_actual: array - True values.
        y_predicted: array - Predicted values.
        model_name: str - Name of the model.

    Returns:
        dict: Dictionary containing calculated metrics.
    """
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    
    metrics = {
        f'{model_name} MSE': mse,
        f'{model_name} RMSE': rmse,
        f'{model_name} MAE': mae,
        f'{model_name} R2 score': r2
    }

    return metrics


def plot_actual_vs_predicted(y_actual, y_predicted, title='', predicted_color='limegreen', true_color='orange'):
    """
    Plot the actual vs predicted values along with a perfect prediction line.

    Parameters:
        y_actual: array - Actual values.
        y_predicted: array - Predicted values.
        title: str - Title for the plot.
        predicted_color: The colour of the predicted values.
        true_color: str - The colour of the true values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_predicted, alpha=0.3, label='Actual vs Predicted RUL', color=predicted_color)
    plt.scatter(y_actual, y_actual, alpha=0.3, label='Perfect Prediction', color=true_color)
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()


def print_summary_statistics(values):
    """
    Print summary statistics for the given values.

    Parameters:
        values: array - Values for which to calculate summary statistics.
    """
    print(f'Minimum value: {np.min(values)}')
    print(f'Mean: {np.mean(values)}')
    print(f'Median: {np.median(values)}')
    print(f'Standard Deviation: {np.std(values)}')
    print(f'Maximum value: {np.max(values)}')
    
    
def plot_distribution(values, title='', xlabel='Values', ylabel='Count', color='skyblue'):
    """
    Plot the distribution of values.

    Parameters:
        values: array-like - Values to be plotted.
        title: str - Title for the plot (default: '').
        xlabel: str - Label for the x-axis (default: 'Values').
        ylabel: str - Label for the y-axis (default: 'Count').
        color: str - Color of the bars in the plot (default: 'skyblue').
    """
    # Count the occurrences of each unique value in the input
    counter = Counter(values)
    unique_values = list(counter.keys())
    counts = list(counter.values())

    # Plot the distribution
    plt.figure(figsize=(12, 6))
    plt.bar(unique_values, counts, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
    
def plot_histogram(y_true, y_pred):
    """
    Plot histograms comparing actual and predicted values.

    Parameters:
    - y_true: array - Actual values.
    - y_pred: array - Predicted values.
    """
    fig = plt.figure(figsize=(6, 5))

    plt.hist([y_true, y_pred], color=['green', 'blue'], edgecolor='black', label=['Actual', 'Predicted'])
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()
