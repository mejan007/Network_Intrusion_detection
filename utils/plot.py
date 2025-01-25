import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import glob
import pandas as pd

def get_unique_filename(out_path, base_name="data_dist_KDD.png"):
    """Generate a unique filename by checking existing files."""
    dist_path = os.path.join(out_path, base_name)
    
    if os.path.exists(dist_path):
        existing_files = glob.glob(os.path.join(out_path, "data_dist_KDD_*.png"))
        numbers = [int(f.split("_")[-1].split(".")[0]) for f in existing_files if f.split("_")[-1].split(".")[0].isdigit()]
        next_num = max(numbers) + 1 if numbers else 1
        dist_path = os.path.join(out_path, f"data_dist_KDD_{next_num}.png")

    return dist_path


def plot_data_distribution(y_train, y_valid, y_test, out_path, lookup):
    """
    Plot class distributions for training, validation, and test sets.

    Args:
    - y_train, y_valid, y_test: NumPy arrays containing numeric class labels.
    - out_path: Directory where the plot should be saved.
    - lookup: Dictionary mapping numeric labels to class names.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Convert numeric labels to class names
    y_train_mapped = np.vectorize(lookup.get)(y_train)
    y_valid_mapped = np.vectorize(lookup.get)(y_valid)
    y_test_mapped = np.vectorize(lookup.get)(y_test)

    # Convert to DataFrames for Seaborn compatibility
    train_df = pd.DataFrame({"Class": y_train_mapped})
    valid_df = pd.DataFrame({"Class": y_valid_mapped})
    test_df = pd.DataFrame({"Class": y_test_mapped})

    # Training set
    sns.countplot(data=train_df, x="Class", ax=axes[0], hue="Class", palette="Set2", legend=False)
    axes[0].set_title('Training Set Class Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Frequency')

    # Validation set
    sns.countplot(data=valid_df, x="Class", ax=axes[1], hue="Class", palette="Set2", legend=False)
    axes[1].set_title('Validation Set Class Distribution')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Frequency')

    # Test set
    sns.countplot(data=test_df, x="Class", ax=axes[2], hue="Class", palette="Set2", legend=False)
    axes[2].set_title('Test Set Class Distribution')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Frequency')

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()

    # Save the plot
    dist_path = get_unique_filename(out_path)
    plt.savefig(dist_path)
    print(f"Saved distribution plot as {dist_path}")

def plot_roc_curve(fpr, tpr, roc_auc, dataset_type, output_folder):
    """Plots and saves the ROC curve."""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset_type}')
    plt.legend(loc="lower right")
    
    # Save the ROC curve
    roc_path = os.path.join(output_folder, f"roc_curve_{dataset_type}.png")
    plt.savefig(roc_path)
    plt.close()
    
def plot_combined_roc_curve(all_fpr, all_tpr, all_auc, k, output_folder):
    """Plots and saves the combined ROC curve for all folds."""
    plt.figure()
    for i in range(k):
        plt.plot(all_fpr[i], all_tpr[i], lw=2, label=f'Fold {i+1} (AUC = {all_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Combined ROC Curve for {k}-Fold Cross Validation')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_folder, f'combined_roc_curve_{k}_fold.png'))
    plt.close()
    
    
def plot_threshold_metrics(thresholds, metrics, metric_name, data_type, output_folder):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, metrics, color='blue', marker='o')
    plt.xlabel('Threshold')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs. Threshold ({data_type})')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'{metric_name}_vs_threshold_{data_type}.png'))
    plt.close()
    
def save_classification_reports_for_thresholds(thresholds, y_true, y_prob, output_folder, prefix):
    """Generate and save classification reports and confusion matrices for different thresholds in a single file."""
    report_file_path = os.path.join(output_folder, f'{prefix}_report_thresholds.txt')
    
    with open(report_file_path, "w") as f:
        for threshold in thresholds:
            # Generate binary predictions based on the threshold
            y_pred = (y_prob >= threshold).astype(int)
            
            # Generate classification report
            report = classification_report(y_true, y_pred)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Write the classification report to the file
            f.write(f"Classification Report for Threshold {threshold:.2f}:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(cm, separator=', '))
            f.write("\n" + "="*50 + "\n")  # Add a separator for readability

    print(f"Classification reports and confusion matrices saved to {report_file_path}")
    
    
def save_loss_curve(losses, output_folder, title):
    """Saves the loss curve over epochs to a file."""
    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f'{title.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()  # Close the figure to prevent display
    
    
def save_combined_loss_curve(all_fold_losses, final_loss, output_folder):
    """Saves the combined loss curves for all folds and final model in one plot."""
    plt.figure()

    # Plot the loss curve for each fold
    for fold_index, losses in enumerate(all_fold_losses):
        plt.plot(losses, label=f'Fold {fold_index + 1} Loss')

    # Plot the final model's loss curve
    plt.plot(final_loss, label='Final Model Loss', linestyle='--', color='black', linewidth=2)

    plt.title('Combined Loss Curves for K-Folds and Final Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot as a single file
    combined_loss_curve_path = os.path.join(output_folder, 'combined_loss_curve.png')
    plt.savefig(combined_loss_curve_path)
    plt.close()  # Close the plot to avoid displaying it in Jupyter
