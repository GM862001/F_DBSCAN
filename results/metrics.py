import sklearn.metrics as mtr
import numpy as np
import bcubed

def ARI_score(true_labels, predicted_labels):
    return mtr.adjusted_rand_score(true_labels, predicted_labels)

def AMI_score(true_labels, predicted_labels):
    return mtr.adjusted_mutual_info_score(true_labels, predicted_labels)

def PURITY_score(true_labels, predicted_labels):
    contingency_matrix = mtr.cluster.contingency_matrix(true_labels, predicted_labels)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def BCubed_Precision_score(true_labels, predicted_labels):
    ldict = {}
    cdict = {}
    for i in range(len(true_labels)):
        ldict[i] = set([true_labels[i]])
        cdict[i] = set([predicted_labels[i]])
    return bcubed.precision(cdict, ldict)

def BCubed_Recall_score(true_labels, predicted_labels):
    ldict = {}
    cdict = {}
    for i in range(len(true_labels)):
        ldict[i] = set([true_labels[i]])
        cdict[i] = set([predicted_labels[i]])
    return bcubed.recall(cdict, ldict)

def print_metrics(true_labels, elaborated_labels, message):
    print(f'{message}:')
    print(f'Purity: {PURITY_score(true_labels, elaborated_labels):.4f}')
    print(f'ARI: {ARI_score(true_labels, elaborated_labels):.4f}')
    print(f'AMI: {AMI_score(true_labels, elaborated_labels):.4f}')
    print(f'BCubed Precision: {BCubed_Precision_score(true_labels, elaborated_labels):.4f}')
    print(f'BCubed Recall: {BCubed_Recall_score(true_labels, elaborated_labels):.4f}\n')