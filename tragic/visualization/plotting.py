import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_attention_maps(attention_maps, save_path=None):
    """Plot attention maps from different layers and heads."""
    num_layers = len(attention_maps)
    num_heads = len(attention_maps[0])
    
    fig, axes = plt.subplots(num_layers, num_heads, 
                            figsize=(4*num_heads, 4*num_layers))
    if num_layers == 1:
        axes = [axes]
    
    for i, layer_maps in enumerate(attention_maps):
        for j, attention in enumerate(layer_maps):
            sns.heatmap(attention, ax=axes[i][j], cmap='viridis')
            axes[i][j].set_title(f'Layer {i+1}, Head {j+1}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels if labels else 'auto',
                yticklabels=labels if labels else 'auto')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(fprs, tprs, auc_scores, title="ROC Curve", save_path=None):
    """Plot ROC curve with confidence intervals."""
    mean_fpr = np.linspace(0, 1, 100)
    tprs_interp = []
    
    plt.figure(figsize=(8, 8))
    
    # Plot individual ROC curves
    for fpr, tpr in zip(fprs, tprs):
        plt.plot(fpr, tpr, 'b-', alpha=0.1)
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
    
    # Calculate mean and std
    mean_tpr = np.mean(tprs_interp, axis=0)
    std_tpr = np.std(tprs_interp, axis=0)
    
    # Plot mean ROC curve
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    plt.plot(mean_fpr, mean_tpr, 'b-',
             label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
    
    # Plot confidence intervals
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, 
                     color='b', alpha=0.2)
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series_examples(X, y, unique_labels, save_path=None):
    """Plot example time series from each class."""
    plt.figure(figsize=(12, 8))
    num_classes = len(unique_labels)
    examples_per_class = min(3, len(X)//num_classes)
    
    for i, cls in enumerate(unique_labels):
        class_indices = np.where(y == i)[0]
        chosen = np.random.choice(class_indices, size=examples_per_class, replace=False)
        for j, idx in enumerate(chosen):
            plt.subplot(num_classes, examples_per_class, i*examples_per_class + j + 1)
            plt.plot(X[idx], color='blue')
            plt.title(f'Class: {cls}', fontsize=12)
            if j == 0:
                plt.ylabel('Value')
            if i == num_classes-1:
                plt.xlabel('Time')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
