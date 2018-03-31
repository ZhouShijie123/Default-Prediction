
# coding: utf-8

# In[1009]:


#code running under python3.x version
#author:yang


# In[1010]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import StratifiedShuffleSplit,StratifiedKFold,train_test_split
from scipy import stats
from sklearn import decomposition,linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
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
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
#import xgboost as xgb
#from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
#import lightgbm as lgb
#from wordcloud import WordCloud
import re
from nltk.corpus import stopwords,wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import os
from collections import Counter
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
get_ipython().magic('matplotlib inline')
#using matplotlib’s ggplot style
plt.style.use('ggplot')
import seaborn as sns
sns.set(color_codes=True)
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import math
import matplotlib as mpl
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, Activation, LSTM, Flatten, Convolution1D, GRU, MaxPooling1D


# In[1011]:


import plotly.graph_objs as go


# In[1012]:


from datetime import datetime


# In[1013]:


#load dataset
thepath ='/Users/gyang/Desktop/ProjectDataScience/final/'

loandata =pd.read_csv(thepath +'loan.csv')


# In[1014]:


pd.set_option('display.max_columns',None)
loandata.head()


# In[1015]:


df = pd.DataFrame({'Total_LoanAmount' : loandata.groupby(['issue_d'])['loan_amnt'].sum(),
                  'Total_LateFee': loandata.groupby(['issue_d'])['total_rec_late_fee'].sum(),
                  'Total_payamount': loandata.groupby(['issue_d'])['total_pymnt'].sum(),
                  'Total_Lastpayamount': loandata.groupby(['issue_d'])['last_pymnt_amnt'].sum(),
                  'Total_OpenAcc': loandata.groupby(['issue_d'])['total_acc'].sum()}).reset_index()


# In[1016]:


df['issue_d'] = pd.to_datetime(df['issue_d'])
df['year_month']=df['issue_d'].apply(lambda x: str(x)[:7])
df['year']=df['issue_d'].apply(lambda x: str(x)[:4])
df =df.drop(['issue_d'],axis =1)


# In[1017]:


df = df.sort_values(by='year_month',ascending=True)


# In[1018]:


df.head()


# In[1019]:


df2 =pd.DataFrame({'year_payment': df.groupby(['year'])['Total_payamount'].sum(),
                   'year_loan': df.groupby(['year'])['Total_LoanAmount'].sum(),
                  'year_lastpayment': df.groupby(['year'])['Total_Lastpayamount'].sum()}).reset_index()


# In[1021]:


df2


# In[1022]:


print(df.shape)


# In[1023]:


trace_1 = go.Scatter(
                x=df2['year'],
                y=df2['year_loan'],
                name = "Total Loan Amount by Year",
                line = dict(color = '#DF53A7'),
                opacity = 0.8)

trace_2 = go.Scatter(
                x=df2['year'],
                y=df2['year_payment'],
                name = "Total Payment Amount by Year",
                line = dict(color = '#12DA90'),
                opacity = 0.8)


data = [trace_1,trace_2]

layout = dict(
    title = "2007 -2015 Credit Loan Overview (By Year)")

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "2007 -2015 Loan Overview(By Year)")


# In[1024]:


trace_1 = go.Scatter(
                x=df2['year'],
                y=df2['year_lastpayment'],
                name = "Total Last Payment Amount by Year",
                line = dict(color = '#DF53A7'),
                opacity = 0.8)

trace_2 = go.Scatter(
                x=df2['year'],
                y=df2['year_payment'],
                name = "Total Payment Amount by Year",
                line = dict(color = '#2822DA'),
                opacity = 0.8)


data = [trace_1,trace_2]

layout = dict(
    title = "2007 -2015 Credit Payment & Last Payment Overview (By Year)")

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "2007 -2015 Loan Overview(By Year)")


# ### [RGB color Selection](http://www.rapidtables.com/web/color/RGB_Color.htm)

# ### Split data and training model

# In[1025]:


df3 =df[['Total_LoanAmount']]
df3 = df3.astype('float32')


# In[1026]:


df3.head()


# In[1027]:


#use last 12 month for test data

X = df3.values
train, test = X[:-12], X[-12:]


# In[1028]:


# reshape data to scale the point
train = train.reshape(-1, 1)
test = test.reshape(-1, 1)

scaler = StandardScaler()
train_n = scaler.fit_transform(train)
test_n = scaler.transform(test)
X_n = scaler.transform(X)


# In[1029]:


print(train_n.shape,test_n.shape,X_n.shape)


# In[1030]:


def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# In[1031]:


look_back = 3
trainX, trainY = create_dataset(train_n, look_back )
testX, testY = create_dataset(test_n, look_back )


# In[1032]:


print(trainX.shape,trainY.shape)
print(testX.shape,testY.shape)


# In[1033]:


model = Sequential()

#build model

#model.add(LSTM(128, input_dim =(None,look_back), return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(128, return_sequences=False))
#model.add(Dropout(0.2))
#model.add(Dense(3, activation='softmax'))
model.add(Dense(32, input_dim = look_back))
model.add(Activation('linear'))

#model.add(Dense(64))
#model.add(Activation('linear'))

model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('linear'))
opt = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999,
                          epsilon=1e-08, decay=0.0)
#complie model
model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.summary()

#fit the model
history =model.fit(trainX, trainY, epochs=100,validation_data=(testX, testY), batch_size=40, verbose=2,shuffle =False)


# In[1034]:


trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore= model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


# In[1040]:


loss = history.history['loss']
val_loss = history.history['val_loss']
#epochs = range(len(loss))
plt.figure()
plt.plot(loss, 'blue', label='Train Loss')
plt.plot(val_loss, 'orange', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.ylim(ymax = 5,ymin =0)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[1036]:


# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(X)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(X)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(X)-1, :] = testPredict


# In[1037]:


# plot baseline and predictions
def demo_plot(factor =1.0):
    default_figsize = mpl.rcParamsDefault['figure.figsize']
    mpl.rcParams['figure.figsize'] = [val *factor for val in default_figsize]
    
    plt.plot(X_n, label ='real data')
    plt.plot(trainPredictPlot, label ='train data')
    plt.plot(testPredictPlot, label ='test data')
    plt.legend(loc="upper left")
    #plt.figure(figsize=(30,15))
    #plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.title('Real Time Series and Predicted Total Loan Amount (2007-2015)')
    plt.show()


# In[1038]:


for scale in [1.5]:
    demo_plot(scale)

