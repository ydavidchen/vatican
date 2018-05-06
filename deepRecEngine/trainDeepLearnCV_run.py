from trainDeepLearnCV import *

if __name__ == '__main__':
	X_train, X_test, y_train, y_test = loadTrainingData(); 
	bestParams = isFluDeepLearnCV(X_train, y_train); 
	clf = retrainModel(bestParams, X_train, y_train); 
	test_score, test_predLab = evalOnTest(clf, X_test); 
	drawROCandCM(y_test, test_predLab, test_score[: , 1]);

print(clf)
quit();