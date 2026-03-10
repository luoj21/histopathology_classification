import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize


def plot_roc_curve(y_pred_proba, y_test, class_names, output_dir):
    """ Plots multi-class ROC curve
    
    Inputs:
    - y_pred_proba: the predicted probability of being in a certain class
    - y_test: the true labels of the test set
    - class_names: the names of the classes
    
    Outputs:
    None"""
    plt.clf()
    
    y_test_binarized=label_binarize(y_test, classes=np.unique(y_test))
    fpr = {}
    tpr = {}
    thresh ={}
    roc_auc = dict()

    y_pred_proba = np.array(y_pred_proba)

    for i in range(0,len(class_names)):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])   
        plt.plot(fpr[i], tpr[i], linestyle='--', label='%s vs Rest (AUC=%0.2f)'%(class_names[i], roc_auc[i]))

    plt.plot([0,1],[0,1],'b--')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend()
    plt.savefig(f"{output_dir}\\roc_curve.png")


def plot_loss_acc(epochs, train_losses, val_losses, train_accs, val_accs, output_dir):
    """ Plots training and validation loss and accuracy curves
    
    Inputs:
    - epochs: list of epoch numbers
    - train_losses: list of training losses
    - val_losses: list of validation losses
    - train_accs: list of training accuracies
    - val_accs: list of validation accuracies
    
    Outputs:
    None"""
    plt.clf()
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}\\loss_acc_curves.png")


def plot_confusion_matrix(y_pred, y_true, class_names, output_dir):
    """ Plots confusion matrix
    
    Inputs:
    - y_pred: the predicted labels
    - y_true: the true labels
    - class_names: the names of the classes
    
    Outputs:
    None"""
    
    plt.clf()
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.savefig(f"{output_dir}\\confusion_matrix.png")