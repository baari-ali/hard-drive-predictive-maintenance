import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) from scratch.

    Parameters:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        mse (float): Mean Squared Error.
    """
    squared_diff = (y_true - y_pred) ** 2
    mse = np.mean(squared_diff)
    return mse


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) from scratch.

    Parameters:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        rmse (float): Root Mean Squared Error.
    """    
    squared_diff = (y_true - y_pred) ** 2
    rmse = np.sqrt(np.mean(squared_diff))
    return rmse


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) from scratch.

    Parameters:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        mae (float): Mean Absolute Error.
    """
    absolute_diff = np.abs(y_true - y_pred)
    mae = np.mean(absolute_diff)
    return mae


def calculate_r2(y_true, y_pred):
    """
    Calculates the R^2 score from scratch.

    Parameters:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        r2 (float): R-squared score.
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
        metrics (dict): Dictionary containing calculated metrics.
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


def plot_distribution(values, 
                      title='', 
                      xlabel='RUL', 
                      ylabel='Count', 
                      color='skyblue', 
                      save_path=None):
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

    plt.figure(figsize=(10, 6))
    plt.bar(unique_values, counts, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.xticks(np.arange(0, 88, 10))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def plot_actual_vs_predicted(y_true, 
                             y_pred, 
                             title='', 
                             true_color='orange', 
                             predicted_color='limegreen', 
                             save_path=None):
    """
    Plot a scatter plot of the actual vs predicted values along with a perfect prediction line.

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
    plt.xlabel('Actual RUL (days)')
    plt.ylabel('Predicted RUL (days)')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.xticks(np.arange(0, 88, 10))
    plt.yticks(np.arange(0, 88, 10))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def plot_summary_statistics(y_true, 
                            y_pred, 
                            title='',
                            true_color='orange', 
                            pred_color='limegreen', 
                            text_color_t='darkorange', 
                            text_color_p='green', 
                            save_path=None):
    """
    Creates a boxplot comparing the summary statistics of the true and predicted values.
    
    Parameters:
        y_true: Actual values.
        y_pred: Predicted values.
        true_color (str): Color for the true boxplot.
        pred_color (str): Color for the predicted boxplot and text annotations. Default is 'limegreen'.
        text_color_t (str): Color the the true text annotations.
        text_color_p (str): Color for the pred text annotations.
        save_path (str): File path to save the figure.
    """
    plt.figure(figsize=(8, 6))
    
    # Calculate summary statistics for y_true
    true_minimum = np.min(y_true)
    true_mean = np.mean(y_true)
    true_median = np.median(y_true)
    true_std_dev = np.std(y_true)
    true_maximum = np.max(y_true)
    
    # Calculate summary statistics for y_pred
    pred_minimum = np.min(y_pred)
    pred_mean = np.mean(y_pred)
    pred_median = np.median(y_pred)
    pred_std_dev = np.std(y_pred)
    pred_maximum = np.max(y_pred)
    
    # Plot boxplot for both y_true and y_pred
    true_boxes = plt.boxplot([y_true], vert=False, positions=[1], labels=['True'], patch_artist=True)
    pred_boxes = plt.boxplot([y_pred], vert=False, positions=[1.3], labels=['Predicted'], patch_artist=True)
    
    # Set colors for the boxplots
    for box in true_boxes['boxes']:
        box.set(facecolor=true_color)
    for box in pred_boxes['boxes']:
        box.set(facecolor=pred_color)
    
    # Annotate summary statistics for y_true
    plt.text(1, 0.85, f'Statistics of true values:', fontsize='medium', color=text_color_t)
    plt.text(1, 0.75, f'Maximum: {true_maximum:.2f}', fontsize='medium', color=text_color_t)
    plt.text(1, 0.7, f'Mean: {true_mean:.2f}', fontsize='medium', color=text_color_t)
    plt.text(1, 0.65, f'Median: {true_median:.2f}', fontsize='medium', color=text_color_t)
    plt.text(1, 0.6, f'Std dev: {true_std_dev:.2f}', fontsize='medium', color=text_color_t)
    plt.text(1, 0.55, f'Minimum: {true_minimum:.2f}', fontsize='medium', color=text_color_t)
    
    # Annotate summary statistics for y_pred
    plt.text(1, 1.75, f'Statistics of predicted values:', fontsize='medium', color=text_color_p)
    plt.text(1, 1.65, f'Maximum: {pred_maximum:.2f}', fontsize='medium', color=text_color_p)
    plt.text(1, 1.6, f'Mean: {pred_mean:.2f}', fontsize='medium', color=text_color_p)
    plt.text(1, 1.55, f'Median: {pred_median:.2f}', fontsize='medium', color=text_color_p)
    plt.text(1, 1.5, f'Std dev: {pred_std_dev:.2f}', fontsize='medium', color=text_color_p)
    plt.text(1, 1.45, f'Minimum: {pred_minimum:.2f}', fontsize='medium', color=text_color_p)
        
    plt.xlabel('RUL (days)')
    plt.title(title)
    plt.grid(True)
    plt.xticks(np.arange(0, 88, 10))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    
def plot_histograms_parallel(y_true, 
                             y_pred, 
                             titles=('Distribution of True RUL', 'Distribution of Predicted RUL'), 
                             xlabels=('True RUL (days)', 'Predicted RUL (days)'), 
                             ylabel='No. of Occurrences', 
                             colors=('orange', 'limegreen'),
                             save_path=None):
    """
    Plot the true and predicted histograms side by side.

    Parameters:
        y_true: Actual RUL values for the first plot.
        y_pred: Predicted RUL values for the second plot.
        titles (tuple): Titles of the plots.
        xlabels (tuple): Labels for the x-axis.
        ylabel (str): Label for the y-axis.
        colors (tuple): Colors for the plots.
        save_path (str): File path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, values in enumerate((y_true, y_pred)):
        ax = axes[i]
        title = titles[i]
        
        ax.hist(values, bins=30, color=colors[i], edgecolor='black', alpha=0.7)
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.set_xticks(np.arange(min(values), max(values)+1, 10))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_histograms_stacked(y_true, 
                            y_pred, 
                            title='Comparison of True and Predicted Values',
                            xlabel='RUL (days)',
                            ylabel='No. of Occurrences',
                            actual_color='orange', 
                            predicted_color='limegreen', 
                            save_path=None):
    """
    Plot 2 histograms with the predictions over the true values for a more detailed comparison.

    Parameters:
        y_true: Actual values.
        y_pred: Predicted values.
        title (str): Title of the plot.
        xlabel (str): Label for thex-axis.
        ylabel (str): Label for the y-axis.
        actual_color (tuple): The histogram color representing the actual values.
        predicted_color (tuple): The histogram color representing the predicted values.
        save_path (str): File path to save the figure.
    """
    true_range = (min(y_true), max(y_true))
    pred_range = (min(y_pred), max(y_pred))
    common_range = (min(true_range[0], pred_range[0]), min(true_range[1], pred_range[1]))

    fig = plt.figure(figsize=(8, 6))

    plt.hist(y_true, bins=30, range=common_range, color=actual_color, alpha=0.5, edgecolor='black', label='Actual')
    plt.hist(y_pred, bins=30, range=common_range, color=predicted_color, alpha=0.5, edgecolor='black', label='Predicted')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(np.arange(common_range[0], common_range[1]+1, 10))
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
