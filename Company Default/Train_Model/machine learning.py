import pandas as pd
import numpy as np
from sklearn import feature_extraction
from scipy import stats
from sklearn import decomposition,linear_model
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier,Lasso,SGDClassifier,LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,hamming_loss
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import pandas as pd
import numpy
import nltk
import re
import os
from nltk.corpus import stopwords,wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn import decomposition
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('data.csv')



theLUT = pd.read_csv('classifierLUT.csv',index_col=0) #ALGO LUT
def optFunc(theAlgo,theParams):
    theModel = theLUT.loc[theAlgo,'optimizedCall']
    tempParam = list()
    for key, value in theParams.iteritems():
        tempParam.append(str(key) + "=" + str(value)) 
    theParams = ",".join(tempParam)
    theModel = theModel + theParams + ")"
    return theModel 

def algoArray(theAlgo):
    theAlgoOut = theLUT.loc[theAlgo,'functionCall']
    return theAlgoOut

train = data.sample(int(len(data) * 0.7))
test = data.drop(train.index)
train_X = train.drop(train.columns[-1],axis=1).values
train_Y = train[train.columns[-1]].values

    

test_X = test.drop(test.columns[-1],axis=1).values
test_Y = test[test.columns[-1]].values

    

theModels = ['RF']#,'NN','DT','LDA'] #these MUST match up with names from LUT #ABDT, #GBC, #RSM take far too long
theResults = pd.DataFrame(0,index=theModels,columns=['accuracy','confidence','runtime'])
for theModel in theModels:
    startTime = time.time()
    model = eval(algoArray(theModel))
    #model = RandomForestClassifier(random_state=50)
    print(theModel)

    #cross validation    
    cvPerf = cross_val_score(model,train_X,train_Y,cv=10)
    theResults.ix[theModel,'accuracy'] = round(cvPerf.mean(),3)
    theResults.ix[theModel,'confidence'] = round(cvPerf.std() * 2,2)
    endTime = time.time()
    theResults.ix[theModel,'runtime'] = round(endTime - startTime,0)
    
print(theResults)

#############################################
#######Run with best performing model########
#####Fine Tune Algorithm Grid Search CV######
#############################################
bestPerfStats = theResults.loc[theResults['accuracy'].idxmax()]
modelChoice = theResults['accuracy'].idxmax()
              
startTime = time.time()
model = eval(algoArray(modelChoice))
grid = eval(theLUT["gridSearch"][modelChoice])
grid.fit(train_X,train_Y)

bestScore = round(grid.best_score_,4)
parameters = grid.best_params_
endTime = time.time()
print("Best Score: " + str(bestScore) + " and Grid Search Time: " + str(round(endTime - startTime,0)))

############################################
######Train Best Model on Full Data Set#####
########Save Model for future use###########
############################################
startTime = time.time()
model = eval(optFunc(modelChoice,parameters)) #train fully validated and optimized model
model.fit(train_X,train_Y)
#model.fit(train,trainIndex)
joblib.dump(model, modelChoice + '.pkl') #save model
endTime = time.time()
print("Model Save Time: " + str(round(endTime - startTime,0)))