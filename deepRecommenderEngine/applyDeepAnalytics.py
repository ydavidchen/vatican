__author__ = "David Chen"
__credit__ = ["David Chen","Won Murdocq","Derrick Wilson-Duncan"]
__copyright__ = "Copyright 2018"
__license__ = "GPL3.0"
__version__ = "1.0.0"
__status__ = "Production"

from trainDeepLearnCV import *
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.figsize":(4,8),
            "font.size":20,
            "axes.titlesize":20,
            "axes.labelsize":25,
            'legend.fontsize':20, 
            "lines.linewidth":4, 
            "xtick.major.size":10,
            "ytick.major.size":10},
        style="whitegrid");

# Load the model from disk and apply to new data:
def loadAndApply():
	'''
	Loads (re)-trained model and apply to new patient data
	'''
	clf = pickle.load(open(filename, 'rb')); 
	newData = pd.read_csv("patient.csv", header=None);
	return clf.predict_proba(newData);

## Data visualization and export for display in HTML:
def visualizeAndExport(probScore, filePath, savefig=True):
    '''
    Visulizes probability of getting flu (barplot) and recommends (title):
    '''
    ## Binary decision
    condition = probScore > 0.50; 
    title = "Should you get a flu shot? \n";
    title += "Yes!" if condition else "No!";

    ## Data visualization
    plt.plot();
    plt.bar(1, probScore, width=2); 
    plt.yticks(np.arange(0,1,0.2)); 
    plt.xticks([]); 
    plt.ylabel('Probability')
    plt.ylim(0, 1);
    plt.title(title);

    if savefig:
        plt.savefig(fname=filePath, dpi=300);
    else:
        plt.show(); 

    return None; 



