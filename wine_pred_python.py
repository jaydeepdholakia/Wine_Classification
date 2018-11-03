
# coding: utf-8

# ## Importing Libraries

# In[1]:

#Library to work with Data:
import pandas as pd

#Bellow all are the Diffrent Models (All Supervised, Obviously!!)

#Our model:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Used for splitting our data into traing and testing sets:
from sklearn import cross_validation

#Used for representing how well did your model do on the validation(Testing data):
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# ## Loading Data

# In[2]:

df = pd.read_csv('data.csv')


# In[3]:

df.head()


# In[4]:

print("Dataset has a shape:")
print(df.shape)


# In[5]:

print("Statistical Summary:")
df.describe()


# In[6]:

print("Class Distribution:")
df.groupby('class').size()


# In[7]:

array = df.values
X = array[:,1:13]
y = array[:,0]
seed = 7
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=seed)


# In[8]:

print("X Data type: ",X.dtype)
print("Y Data type: ",y.dtype)


# In[9]:

print("X_train Data type: ",X_train.dtype)
print("X_test Data type: ",X_test.dtype)
print("y_train Data type: ",y_train.dtype)
print("y_test Data type: ",y_test.dtype)


# In[10]:

print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_train.shape)


# ## Creating & Testing Model

# In[11]:

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
predict = model.predict(X_test)
print(accuracy_score(y_test, predict))
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))

