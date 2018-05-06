__author__ = "David Chen"
__credit__ = ["David Chen","Won Murdocq","Derrick Wilson-Duncan"]
__copyright__ = "Copyright 2018"
__license__ = "GPL3.0"
__version__ = "1.0.0"
__status__ = "Production"

from trainDeepLearnCV import *
import pickle

if __name__ == '__main__':
	X_train, X_test, y_train, y_test = loadTrainingData(); 
	bestParams = isFluDeepLearnCV(X_train, y_train); 
	clf = retrainModel(bestParams, X_train, y_train); 
	test_score, test_predLab = evalOnTest(clf, X_test); 
	drawROCandCM(y_test, test_predLab, test_score[: , 1]);
	filename = 'retrainedFinalModel.sav'; 
	pickle.dump(clf, open(filename, 'wb'))
print(clf)
quit();