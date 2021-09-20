# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score


def optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for classification
    ----------
    target: true labels
    predicted: positive probability predicted by the model.
    i.e. model.prdict_proba(X_test)[:, 1], NOT 0/1 prediction array
    Returns
    -------     
    cut-off value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    
    return round(list(roc_t['threshold'])[0], 2)

def plot_confusion_matrix(y_true, y_pred):
    # confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    data = conf_matrix.transpose()  
    
    _, ax = plt.subplots()
    ax.matshow(data, cmap="Blues")
    # printing exact numbers
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{}'.format(z), ha='center', va='center')
    # axis formatting 
    plt.xticks([])
    plt.yticks([])
    plt.title("True label\n 0  {}     1\n".format(" "*18), fontsize=14)
    plt.ylabel("Predicted label\n 1   {}     0".format(" "*18), fontsize=14)
    
def draw_roc_curve(y_true, y_proba):
    '''
    y_true: 0/1 true labels for test set
    y_proba: model.predict_proba[:, 1] or probabilities of predictions
    
    Return:
        ROC curve with appropriate labels and legend 
    
    '''
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    _, ax = plt.subplots()
    
    ax.plot(fpr, tpr, color='r');
    ax.plot([0, 1], [0, 1], color='y', linestyle='--')
    ax.fill_between(fpr, tpr, label=f"AUC: {round(roc_auc_score(y_true, y_proba), 3)}")
    ax.set_aspect(0.90)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(-0.02, 1.02);
    ax.set_ylim(-0.02, 1.02);
    plt.legend()
    plt.show()


def summerize_results(y_true, y_pred):
    '''
     Takes the true labels and the predicted probabilities
     and prints some performance metrics.
    '''
    print("\n=========================")
    print("        RESULTS")
    print("=========================")

    print("Accuracy: ", accuracy_score(y_true, y_pred).round(2))
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = round(conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[1, 0]), 2)
    specificity = round(conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[0, 1]), 2)
    
    ppv = round(conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[0, 1]), 2)
    npv = round(conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[1, 0]), 2)
    
    print("-------------------------")
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    
    print("-------------------------")
    
    print("positive predictive value: ", ppv)
    print("negative predictive value: ", npv)
    
    print("-------------------------")
    print("precision: ", precision_score(y_true, y_pred).round(2))
    print("recall: ", recall_score(y_true, y_pred).round(2))


