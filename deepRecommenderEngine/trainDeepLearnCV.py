__author__ = "David Chen"
__credit__ = ["David Chen","Won Murdocq","Derrick Wilson-Duncan"]
__copyright__ = "Copyright 2018"
__license__ = "GPL3.0"
__version__ = "1.0.0"
__status__ = "Production"

import numpy as np
import pandas as pd
import itertools
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, accuracy_score, make_scorer

def loadTrainingData(testFrac=0.10):
	'''
	Loads US-CDC training data and split into training and test set
	:param testFrac: Fraction used for testing. Defaults to 10%.
	'''
	## Load 3-year data:
	df17 = pd.read_csv("data/2017VAERSDATA.csv", index_col=None).merge(pd.read_csv("data/2017VAERSVAX.csv", index_col=None), on="VAERS_ID");
	df16 = pd.read_csv("data/2016VAERSDATA.csv", index_col=None).merge(pd.read_csv("data/2016VAERSVAX.csv", index_col=None), on="VAERS_ID");
	df15 = pd.read_csv("data/2015VAERSDATA.csv", index_col=None).merge(pd.read_csv("data/2015VAERSVAX.csv", index_col=None), on="VAERS_ID");

	## Combine into a single data set:
	df = df17.copy(); 
	df = df.append(df16);
	df = df.append(df15); 

	## Exclude records with 0 counts:
	df = df[pd.notnull(df['VAX_TYPE'])];
	df = df[df.VAX_TYPE != 'UNK']; #drop unknowns

	## Convert to date-time format:
	df['RECVDATE'] = pd.to_datetime(df['RECVDATE']); 

	## Categorize to 4 seasons:
	df.loc[(df.RECVDATE >= '12-21-16') & (df.RECVDATE < '03-20-17'), 'Season'] = "Winter"; 
	df.loc[(df.RECVDATE >= '03-20-17') & (df.RECVDATE < '06-21-17'), 'Season'] = "Spring"; 
	df.loc[(df.RECVDATE >= '06-21-17') & (df.RECVDATE < '09-22-17'), 'Season'] = "Summer";
	df.loc[(df.RECVDATE >= '09-22-17') & (df.RECVDATE < '12-21-17'), 'Season'] = "Fall";

	df.loc[(df.RECVDATE >= '12-21-15') & (df.RECVDATE < '03-20-16'), 'Season'] = "Winter"; 
	df.loc[(df.RECVDATE >= '03-20-16') & (df.RECVDATE < '06-21-16'), 'Season'] = "Spring"; 
	df.loc[(df.RECVDATE >= '06-21-16') & (df.RECVDATE < '09-22-16'), 'Season'] = "Summer";
	df.loc[(df.RECVDATE >= '09-22-16') & (df.RECVDATE < '12-21-16'), 'Season'] = "Fall";

	df.loc[(df.RECVDATE >= '12-21-14') & (df.RECVDATE < '03-20-15'), 'Season'] = "Winter"; 
	df.loc[(df.RECVDATE >= '03-20-15') & (df.RECVDATE < '06-21-15'), 'Season'] = "Spring"; 
	df.loc[(df.RECVDATE >= '06-21-15') & (df.RECVDATE < '09-22-15'), 'Season'] = "Summer";
	df.loc[(df.RECVDATE >= '09-22-15') & (df.RECVDATE < '12-21-15'), 'Season'] = "Fall";

	## Assign the dependent variable to a new column:
	df['Vax_group'] = df.VAX_TYPE;
	df.loc[ [bool(re.search("FLU", i)) for i in df.VAX_TYPE], 'Vax_group'] = 'FLU';
	df['isFlu'] = 1 * ( df['Vax_group'] == 'FLU' ); #ichotomize

	## Select features and exclude rows with all missing values:
	df = df[['AGE_YRS','SEX','Season','STATE','isFlu']].dropna(how='all'); 

	## Implement train-test split & export:
	y = df['isFlu']
	X = df.loc[:, df.columns != 'isFlu'].apply(LabelEncoder().fit_transform);
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testFrac); 
	return X_train, X_test, y_train, y_test; 

def isFluDeepLearnCV(X_train, y_train, nfold=10, 
					 alphaRange=10.0 ** -np.arange(1,5), initRange=10.0 ** -np.arange(1,5),
					 layarRange=np.arange(100,200,50), nLayer=100, solverList=['adam','lbfgs']):
	'''
	Implements cross validation & grid search on CDC data with constant layers
	'''
	hiddenLayers = []; 
	for i in layarRange:
	    hiddenLayers += [x for x in itertools.product((nLayer, ), repeat=i)]; 

	hyperParams = {'solver':solverList, 'alpha':alphaRange, 'hidden_layer_sizes':hiddenLayers, 'learning_rate_init':initRange};
	clf_grid = GridSearchCV(MLPClassifier(activation='logistic'), cv=nfold, param_grid=hyperParams, n_jobs=-1);
	clf_grid.fit(X_train, y_train);

	bestValues = clf_grid.best_params_;
	print("The best grid-search score is: %0.3f" % clf_grid.best_score_)
	return bestValues; 

def retrainModel(bestValues, X_train, y_train):
	clf = MLPClassifier(activation='logistic',
						solver = bestValues['solver'],
						learning_rate_init=bestValues['learning_rate_init'],
						alpha = bestValues['alpha'],
						hidden_layer_sizes= bestValues['hidden_layer_sizes']); 
	clf.fit(X_train, y_train);
	return clf; 

def evalOnTest(clf, X_test):
	predScore = clf.predict_proba(X_test);
	predClass = clf.predict(X_test);
	return predScore, predClass; 


import matplotlib.pyplot as plt

def drawROCandCM(y_true, y_pred, y_score, classes=[0,1], 
    fontSize=10, title=None, rot=0, lw=3, rocFigSize=(8,8),
    showConfusion=True, showROC=True, showColorBar=False):
    '''
    Sketches ROC curves and confusion matrix (heat map)
    '''
    n = len(y_true);
    conf = confusion_matrix(y_true, y_pred);
    if showConfusion:
        plotConfusionMatrix(cm=conf, title="Confusion Matrix: "+title, rot=rot, classes=classes)
    acc = float(conf[0, 0] + conf[1, 1]) / n;
    tpr = float(conf[0, 0]) / n;
    tnr = float(conf[1, 1]) / n;
    fpr = float(conf[1, 0]) / n;
    fnr = float(conf[0, 1]) / n;
    f1 = f1_score(y_true, y_pred, average="binary", sample_weight=None); 
    
    rates_fp_roc = dict();
    rates_tp_roc = dict();
    rates_fp_roc, rates_tp_roc, _ = roc_curve(y_true, y_score);
    roc_auc = auc(rates_fp_roc, rates_tp_roc);
    if showROC == True:
        plt.figure(figsize=rocFigSize)
        plt.plot(rates_fp_roc, rates_tp_roc, color='salmon',lw=lw,label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0,1], [0,1], color='gray', lw=lw-1, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig("AUC_curve.png", dpi=300); 
    return None;

def plotConfusionMatrix(cm, classes, title, rot=0, normalize=False,cmap=plt.cm.Blues, showColorBar=False):
    """
    Helper function to display better-looking confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]; 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if showColorBar: 
        plt.colorbar()
    tick_marks = np.arange(len(classes));
    plt.xticks(tick_marks, classes, rotation=rot)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd';
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout();
    plt.xlabel('Predicted Class');
    plt.ylabel('True Class');
    plt.savefig("confusion_matrix.png", dpi=300); 
    return None;

