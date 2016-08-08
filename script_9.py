# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 09:49:27 2016

@author: Tony
"""

import pandas as pd
import numpy as np
np.random.seed(2016)
#import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.cross_validation import train_test_split


datadir = '../input'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                        index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                        index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),
                        parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                        usecols=['event_id','app_id','is_active'],
                        dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))


gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])


brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))

m = phone.phone_brand.str.cat(phone.device_model)
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))


appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())
deviceapps.head()


d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))


applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)


devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelabels.head()


d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0],nlabels))
#Xtr_label = csr_matrix((d['size'], (d.trainrow, d.label)), 
#                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 
                      shape=(gatest.shape[0],nlabels))
#Xte_label = csr_matrix((d['size'], (d.testrow, d.label)), 
#                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))


Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))


targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)


def score(clf, random_state = 0):
    kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
    pred = np.zeros((y.shape[0],nclasses))
    for itrain, itest in kf:
        Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]
        ytr, yte = y[itrain], y[itest]
        clf.fit(Xtr, ytr)
        pred[itest,:] = clf.predict_proba(Xte)
        # Downsize to one fold only for kernels
        return log_loss(yte, pred[itest, :])
        print "{:.5f}".format(log_loss(yte, pred[itest,:]))
    print('')
    return log_loss(y, pred)

#Cs = np.logspace(-2,-1,5)
#res = []
#for C in Cs:
#    res.append(score(LogisticRegression(C = C)))
#plt.semilogx(Cs, res,'-o');

print "LR:{}".format(score(LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')))
#print "LR:{}".format(score(KNeighborsClassifier()))

X_train, X_cv, y_train, y_cv = train_test_split(
    Xtrain, y, train_size=.90, random_state=10)

dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_cv, y_cv)

params = {
    "objective": "multi:softprob",
    "num_class": 12,
    "booster": "gblinear",
    "max_depth":6,
    "eval_metric": "mlogloss",
    "eta": 0.07,
    "silent": 1,
    "alpha":3,
}

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, 40, evals=watchlist,
                early_stopping_rounds=25, verbose_eval=True)

print("# Train")
dtrain = xgb.DMatrix(Xtrain, y)
gbm = xgb.train(params, dtrain, 40, verbose_eval=True)
predXGB = gbm.predict(xgb.DMatrix(Xtest))

clfLR = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
clfLR.fit(Xtrain, y)
predLR = clfLR.predict_proba(Xtest)

num_inputs = X_train.shape[1]
hidden_units_1 = 48
num_classes = 12
p_dropout = 0.0

def indicator(y):
    ind = np.zeros((y.shape[0], num_classes))
    ind[np.arange(y.shape[0]), y] = 1
    return ind

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import Adagrad
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=num_inputs, W_regularizer=l2(1.0)))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=num_classes, W_regularizer=l2(1.0)))
    model.add(Activation("softmax"))
    # Compile model
    opt = Adagrad(lr=0.01, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

class AdaNNClassifier(KerasClassifier):
    def __init__:
    KerasClassifier(build_fn=create_model)

clfAdaNN = AdaBoostClassifier(base_estimator=AdaNNClassifier, n_estimators=20,
                              algorithm='SAMME.R', random_state=0)
clfAdaNN.set_params(AdaNNClassifier__nb_epoch=6, AdaNNClassifier__batch_size=100,
          AdaNNClassifier__verbose=2, 
          AdaNNClassifier__validation_data=(X_cv.toarray(), indicator(y_cv)))
clfAdaNN.fit(X_train.toarray(), y_train)

n=112071 // 20
m=112071 % 20
predNN = np.zeros((112071,12))
for i in xrange(112071):
    predNN[i*20:(i+1)*20,:] = clfAdaNN.predict_proba(Xtest[i*20:(i+1)*20,:].toarray(),verbose=0)
predNN[(i+1)*20:-1,:] = clfAdaNN.predict_proba(Xtest[(i+1)*20:-1,:].toarray(), verbose=0)
#predNN = model.predict_proba(Xtest.toarray(), batch_size=20)

pred = (predNN + predLR + predXGB) / 3.
pred = pd.DataFrame(pred, index = gatest.index, columns=targetencoder.classes_)
pred.head()

pred.to_csv('subm_LR_xgb_NN.csv.gz',index=True, compression="gzip")
