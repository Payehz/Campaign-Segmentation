# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:20:37 2022

@author: User
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

from modules_for_segmentation import EDA, ModelCreation

#%% Statics

CSV_PATH = os.path.join(os.getcwd(),'dataset','Train.csv')
LABEL_ENCODER_PATH = os.path.join(os.getcwd(),'le.pkl')
SCALER_PATH = os.path.join(os.getcwd(),'scaler.pkl')
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'campaign_logs', log_dir)
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

#%% Step 1) Data Loading

df = pd.read_csv(CSV_PATH)

#%% Step 2) Data Inspection

df.head()
df.tail()

df.info()
df.describe().T

df.boxplot()

df.isna().sum()
# customer_age, marital, balance, personal_loan, last_contact_duration,
# num_contacts_in_campaign has some NaN values.
# days_since_prev_campaign_contact has too many NaN.
# Hence will not be selected as features.

df.duplicated().sum() # No duplicated data

# Separate into categorical and continuous features
categorical = ['job_type','marital','education','default','housing_loan',
               'personal_loan','communication_type','day_of_month',
               'month','prev_campaign_outcome','term_deposit_subscribed']

continuous = ['customer_age','balance','last_contact_duration',
              'num_contacts_in_campaign','num_contacts_prev_campaign']

# ID column is ignored

# To visualise data
eda = EDA()

for i in categorical:
    eda.cat_graph_plot(df,i)
    
for j in continuous:
    eda.con_graph_plot(df,j)

for i in categorical:
    plt.figure()
    sns.countplot(df[i],hue=df['term_deposit_subscribed'])
    plt.show()

#%% Step 3) Data Cleaning

# drop ID and days_since_prev_campaign_contact column

df = df.drop(labels=['id','days_since_prev_campaign_contact'],axis=1)

# Label Encoding

le = LabelEncoder()

for cat in categorical:
    temp = df[cat]
    temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
    df[cat] = pd.to_numeric(temp,errors='coerce')

with open(LABEL_ENCODER_PATH,'wb') as file:
    pickle.dump(le,file)
    
# Deal with NaN values

# customer_age, marital, balance, personal_loan, last_contact_duration,
# num_contacts_in_campaign has some NaN values.

# Use KNNImputer

column_names = df.columns

knn_imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')
imputed_data = knn_imputer.fit_transform(df)
df = pd.DataFrame(imputed_data)
df.columns = column_names

for i in column_names:
    df.loc[:,i] = np.floor(df.loc[:,i]).astype('int')

df.info()

#%% Step 4) Features Selection

# categorical = ['job_type','marital','education','default','housing_loan',
#                'personal_loan','communication_type','day_of_month',
#                'month','prev_campaign_outcome','term_deposit_subscribed']

# Categorical vs categorical
# Use Cramer's V

for i in categorical:
    confusionmatrix = pd.crosstab(df[i], df['term_deposit_subscribed']).to_numpy()
    print(i + ": " + str(eda.cramers(confusionmatrix)))

print("------------------")

# continuous = ['customer_age','balance','last_contact_duration',
#               'num_contacts_in_campaign','num_contacts_prev_campaign']

# Continuous vs categorical
# Use logistic regression

for i in continuous:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i],axis=-1),df['term_deposit_subscribed'])
    print(i + ": " + str(lr.score(np.expand_dims(df[i],axis=-1),df['term_deposit_subscribed'])))

# We found out that the accuracy of the model when features selection is not
# done is higher than the accuracy when features selection is done.
# Hence, we ignore the features selection.

#%% Step 5) Data Preprocessing

# Use standard scaler
StdScl = StandardScaler()

# Scalling for continuous features
for con in continuous:
    df[con]= StdScl.fit_transform(np.expand_dims(df[con],axis=-1))

with open(SCALER_PATH,'wb') as file:
    pickle.dump(StdScl,file)

X = df.drop(labels='term_deposit_subscribed',axis=1)
y = df.loc[:,'term_deposit_subscribed']

num_class = len(np.unique(y))

num_feature = np.shape(X)[1:]

# One Hot Encoding
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))

with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X, y,
                                                 test_size = 0.3,
                                                 random_state=123)

#%% Model Development

mc = ModelCreation()

model = mc.simple_two_layer_model(num_class,num_feature)
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')

tensorboard_callback = TensorBoard(log_dir=LOG_FOLDER_PATH)

early_stopping_callback = EarlyStopping(monitor='loss', patience=3)

hist = model.fit(x=X_train,y=y_train,batch_size=128,epochs=100,
                 validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback,early_stopping_callback])

hist.history.keys()
training_loss = hist.history['loss']
training_acc = hist.history['acc']
validation_acc = hist.history['val_acc']
validation_loss = hist.history['val_loss']

# Graph plot
mc.result_plot(training_loss,validation_loss,training_acc,validation_acc)

#%% Saving model

model.save(MODEL_SAVE_PATH)

#%% Plot model

plot_model(model,show_shapes=True,show_layer_names=(True))

#%% Model Evaluation

results = model.evaluate(X_test,y_test)
print(results)

pred_y = np.argmax(model.predict(X_test), axis=1)
true_y = np.argmax(y_test,axis=1)

cm = confusion_matrix(true_y,pred_y)
cr = classification_report(true_y,pred_y)
print(cm)
print(cr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%% Discussion

# The model scores an accuracy of 92%.
# Features selection is ignored since the accuracy when all features are selected
# scores a higher accuracy.
# The model consist of two hidden layers, each with 64 nodes. Dropouts of 20%
# and batch normalizations are also included in the model.
# The model is neither underfitting nor overfitting.









