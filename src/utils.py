import matplotlib.pyplot as plt
import os
from src.constants import *
import pandas as pd
import numpy as np

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

def plot_accuracies(accuracy_list, accuracy_list_val, folder):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    trainAcc_val = [i[0] for i in accuracy_list_val]
    lrs = [i[1] for i in accuracy_list]
    fig, (ax1, ax3) = plt.subplots(2,1, figsize=(7.5,7.5))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Average Training Loss')
    ax1.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle = '-', c='darkblue')
   # ax1.legend(bbox_to_anchor=(0.99,0.9))
    ax1.plot(range(len(trainAcc_val)), trainAcc_val, label='Average Validation Loss', linewidth=1, linestyle = '-', c='firebrick')
    ax1.legend(bbox_to_anchor=(0.99,0.9))
    
    #ax2.set_xlabel('Epochs')
    #ax2.set_ylabel('Average Training Loss')
    #ax2.legend(bbox_to_anchor=(0.99,0.9))
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate')
    ax3.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
    ax3.legend(bbox_to_anchor=(0.91,0.8))
    fig.tight_layout()
    fig.savefig(f'plots/{folder}/training-graph.pdf')
    plt.clf()
    
def cut_array(percentage, arr):
    print(f'{color.BOLD}Slicing dataset to {int(percentage*100)}%{color.ENDC}')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window : mid + window, :]

def getresults2(df, result):
	results2, df1, df2 = {}, df.sum(), df.mean()
	for a in ['FN', 'FP', 'TP', 'TN']:
		results2[a] = df1[a]
	for a in ['precision', 'recall']:
		results2[a] = df2[a]
	results2['f1*'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
	return results2