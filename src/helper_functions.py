import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) from scratch.

    Parameters:
        y_true: array-like, shape (n_samples,) - True values.
        y_pred: array-like, shape (n_samples,) - Predicted values.

    Returns:
        mse: float - MSE.
    """
    squared_diff = (y_true - y_pred) ** 2
    mse = np.mean(squared_diff)
    return mse


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) from scratch.

    Parameters:
        y_true: array-like, shape (n_samples,) - True values.
        y_pred: array-like, shape (n_samples,) - Predicted values.

    Returns:
        rmse: float - RMSE.
    """    
    squared_diff = (y_true - y_pred) ** 2
    rmse = np.sqrt(np.mean(squared_diff))
    return rmse


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) from scratch.

    Parameters:
        y_true: array-like, shape (n_samples,) - True values.
        y_pred: array-like, shape (n_samples,) - Predicted values.

    Returns:
        mae: float - MAE.
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
    if model_name:
        print(f"Model: {model_name}\n")

    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2 score': r2
    }

    return metrics


def plot_distribution(values, title='', xlabel='Values', ylabel='Count', color='skyblue', save_path=None):
    """
    Plot the distribution of RUL values.

    Parameters:
        values: RUL values to be plotted.
        title (str): Title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color (str): Color of the bars in the plot.
        save_path (str): File path to save the figure.
    """
    counter = Counter(values)
    unique_values = list(counter.keys())
    counts = list(counter.values())

    plt.figure(figsize=(10, 5))
    plt.bar(unique_values, counts, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, title='', true_color='orange', predicted_color='limegreen', save_path=None):
    """
    Plot the actual vs predicted values along with a perfect prediction line.

    Parameters:
        y_true: Actual values.
        y_pred: Predicted values.
        title (str): Title for the plot.
        true_color (str): The color of the true values.
        predicted_color (str): The color of the predicted values.
        save_path (str): File path to save the figure.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, label='Actual vs Predicted RUL', color=predicted_color)
    plt.scatter(y_true, y_true, alpha=0.3, label='Perfect Prediction', color=true_color)
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title(title)
    plt.legend(loc='upper left')
    if save_path:
        plt.savefig(save_path)
    plt.show()

    
def summary_statistics(values):
    """
    Calculate summary statistics for the given values.

    Parameters:
        values: Values for which to calculate summary statistics.

    Returns:
        dict: Dictionary containing summary statistics.
            Keys: 'minimum', 'mean', 'median', 'std_dev', 'maximum'
    """
    summary_stats = {
        'minimum': np.min(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std_dev': np.std(values),
        'maximum': np.max(values)
    }
    
    return summary_stats  


def plot_summary_statistics(y_true, y_pred, 
                            titles=('Distribution of Actual RUL', 'Distribution of Predicted RUL'), 
                            xlabels=('Actual RUL', 'Predicted RUL'), 
                            ylabel='Frequency', 
                            colors=('orange', 'limegreen'),
                            save_path=None):
    """
    Visualize summary statistics of actual or predicted RUL values side by side.

    Parameters:
        y_true: Actual RUL values for the first plot.
        y_pred: Predicted RUL values for the second plot.
        titles (tuple): Titles of the plots.
        xlabels (tuple): Labels for the x-axis.
        ylabel (str): Label for the y-axis.
        colors (tuple): Colors for the plots.
        save_path (str): File path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    for i, values in enumerate((y_true, y_pred)):
        ax = axes[i]
        title = titles[i]
        stats = summary_statistics(values)
        minimum = stats['minimum']
        mean = stats['mean']
        median = stats['median']
        std_dev = stats['std_dev']
        maximum = stats['maximum']
        
        ax.hist(values, bins=30, color=colors[i], edgecolor='black', alpha=0.7)
        ax.axvline(minimum, color='grey', linestyle='dashed', linewidth=2, label=f'Minimum: {minimum}')
        ax.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
        ax.axvline(median, color='purple', linestyle='dashed', linewidth=2, label=f'Median: {median}')
        ax.axvline(mean - std_dev, color='blue', linestyle='dashed', linewidth=2, label=f'Standard Deviation: {std_dev:.2f}')
        ax.axvline(mean + std_dev, color='blue', linestyle='dashed', linewidth=2)
        ax.axvline(maximum, color='grey', linestyle='dashed', linewidth=2, label=f'Maximum: {maximum}')
        
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_histogram(y_true, y_pred, 
                   title='Comparison of Actual and Predicted Values',
                   actual_color='orange', predicted_color='limegreen', save_path=None):
    """
    Plot histograms comparing actual and predicted values.

    Parameters:
        y_true: Actual values.
        y_pred: Predicted values.
        actual_color (tuple): The histogram color representing the actual values.
        predicted_color (tuple): The histogram color representing the predicted values.
        save_path (str): File path to save the figure.
    """
    fig = plt.figure(figsize=(8, 6))

    plt.hist(y_true, bins=30, color=actual_color, alpha=0.5, edgecolor='black', label='Actual')
    plt.hist(y_pred, bins=30, color=predicted_color, alpha=0.5, edgecolor='black', label='Predicted')

    plt.title(title)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
