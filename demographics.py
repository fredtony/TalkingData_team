# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 23:14:40 2016

Obviously needs to do better than 1/12 since there are 12 output classes. This
is an 8% accuracy. This means that Log loss would be -(1/m)*m*ln(1/12) = 2.48

@author: Tony
"""

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import cPickle


def logloss(pred_prob, actual):
    # Takes the probabilities of each class and compares them with actual
    # values to give the log loss. Limit values near 0 and 1 since they are
    # undefined. Returns log loss only of actual class.
    pred_prob[pred_prob < 1e-15] = 1e-15
    pred_prob[pred_prob > 1-1e-15] = 1-1e-15
    log_prob = np.log(pred_prob)
    indicator_actual = np.zeros(pred_prob.shape)
    indicator_actual[np.arange(len(actual)), actual] = 1
    err = -np.multiply(log_prob, indicator_actual)
    return err.sum()/err.shape[0]
    
def sklearn_predict_check(clf, X_train, y_train, X_test, y_test):
    # Uses a sklearn classifier that can train probablities on a set and
    # returns matrix of probability predictions for test set
    clf.fit(X_train, y_train)
    # Scores of training on test set & training set
    print "Training set accuracy: {}".format(clf.score(X_train, y_train))
    print "Test set accuracy: {}".format(clf.score(X_test, y_test))
    pred_prob = clf.predict_proba(X_test)
    return pred_prob
    
def xgboost_predict_check(X_train, y_train, X_cv, y_cv, num_round=300):
    # Uses xgboost classifier that can train probablities on a set and
    # returns matrix of probability predictions for test set
    import xgboost as xgb
    
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_cv = xgb.DMatrix(X_cv, label=y_cv)
    
    param = {}
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'mlogloss'
    param['eta'] = 0.03
    param['min_child_weight'] = 3
    param['gamma'] = 0.1
    param['colsample_bytree'] = 0.7
    param['subsample'] = 0.75
    param['max_depth'] = 5
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = 12
    param['random_state'] = 0
    
    watchlist = [(xg_train,'train'), (xg_cv, 'test')]
    clf = xgb.train(param, xg_train, num_round, watchlist)
    
    # get predictions
    pred_prob_train = clf.predict(xg_train).reshape(y_train.shape[0], 12)
    pred_train = np.argmax(pred_prob_train, axis=1)
    
    pred_prob_cv = clf.predict(xg_cv).reshape(y_cv.shape[0], 12)
    pred_cv = np.argmax(pred_prob_cv, axis=1)
    
    train_acc = sum(pred_train == y_train)/float(y_train.shape[0])
    cv_acc = sum(pred_cv == y_cv)/float(y_cv.shape[0])
    
    print "Training set accuracy: {}".format(train_acc)
    print "Test set accuracy: {}".format(cv_acc)
    return pred_prob_cv
    
def xgboost_predict_check_kf(X, y, num_inputs, num_classes, kf_size=10, num_round=300):
    # Uses xgboost classifier that can train probablities on a set and
    # returns matrix of probability predictions for test set
    import xgboost as xgb
    from sklearn.cross_validation import StratifiedKFold
    skf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=0)
    
    param = {}
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'mlogloss'
    param['eta'] = 0.03
    param['min_child_weight'] = 3
    param['gamma'] = 0.1
    param['colsample_bytree'] = 0.7
    param['subsample'] = 0.75
    param['max_depth'] = 5
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = 12
    param['random_state'] = 0
       
    pred_prob_cv = np.zeros((y.shape[0], num_classes))
    for itrain, itest in skf:
        xg_train = xgb.DMatrix(X[itrain], label=y[itrain])
        xg_cv = xgb.DMatrix(X[itest], label=y[itest])
    
        watchlist = [(xg_train,'train'), (xg_cv, 'test')]
        clf = xgb.train(param, xg_train, num_round, watchlist)
    
        # get predictions
#        pred_prob_train = clf.predict(xg_train).reshape(y.shape[0], 12)
#        pred_train = np.argmax(pred_prob_train, axis=1)
        
        pred_prob_cv[itest] = clf.predict(xg_cv).reshape(y[itest].shape[0], num_classes)
    pred_cv = np.argmax(pred_prob_cv, axis=1)
        
#   train_acc = sum(pred_train == y_train)/float(y_train.shape[0])
    cv_acc = sum(pred_cv == y)/float(y.shape[0])
        
    print "CV set accuracy: {}".format(cv_acc)
    return pred_prob_cv

def organize_data(train_size=59872):
    #Used 59872, which is 80%, rounded in a fashion to use large mini-batches that align in size

    with open('dev_df.pkl', 'r') as f:
        dev_df = pd.DataFrame(cPickle.load(f))
    
    # Training/CV set
    gender_age_train = pd.read_csv('gender_age_train.csv', index_col=0).drop(['gender', 'age'], axis=1)
    gender_age_train = gender_age_train.join(dev_df)
    
    # Test set
    gender_age_test = pd.read_csv('gender_age_test.csv', index_col=0)
    gender_age_test = gender_age_test.join(dev_df)
    
    # Labels will be in y array; features will be in X matrix; need to encode labels
    # for phone_brand, device_model, and group
    X = np.array(gender_age_train)
    X_test = np.array(gender_age_test)
    
    # Row 0 is the group to be classified, so put it in y array then delete it
    y = X[:,0]
    from sklearn.preprocessing import LabelEncoder
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    X = np.delete(X,0,1)
    
    # Reformat all labeled columns with label encoders
    le_phone_brand = LabelEncoder()
    le_phone_brand.fit(np.hstack((X[:,0], X_test[:,0])))
    X[:,0] = le_phone_brand.transform(X[:,0])
    X_test[:,0] = le_phone_brand.transform(X_test[:,0])
    
    le_device_model = LabelEncoder()
    le_device_model.fit(np.hstack((X[:,1], X_test[:,1])))
    X[:,1] = le_device_model.transform(X[:,1])
    X_test[:,1] = le_device_model.transform(X_test[:,1])
    
    # Standardize features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(np.vstack((X, X_test)))
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    
    # Create CV set
    from sklearn.cross_validation import train_test_split
    
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, train_size=train_size, random_state=0)
    return X_train, X_cv, y_train, y_cv, X_test
    
def organize_data_kf():
    with open('dev_df.pkl', 'r') as f:
        dev_df = pd.DataFrame(cPickle.load(f))
    
    # Training/CV set
    gender_age_train = pd.read_csv('gender_age_train.csv', index_col=0).drop(['gender', 'age'], axis=1)
    gender_age_train = gender_age_train.join(dev_df)
    
    # Test set
    gender_age_test = pd.read_csv('gender_age_test.csv', index_col=0)
    gender_age_test = gender_age_test.join(dev_df)
    
    # Labels will be in y array; features will be in X matrix; need to encode labels
    # for phone_brand, device_model, and group
    X = np.array(gender_age_train)
    X_test = np.array(gender_age_test)
    
    # Row 0 is the group to be classified, so put it in y array then delete it
    y = X[:,0]
    from sklearn.preprocessing import LabelEncoder
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    X = np.delete(X,0,1)
    
    # Reformat all labeled columns with label encoders
    le_phone_brand = LabelEncoder()
    le_phone_brand.fit(np.hstack((X[:,0], X_test[:,0])))
    X[:,0] = le_phone_brand.transform(X[:,0])
    X_test[:,0] = le_phone_brand.transform(X_test[:,0])
    
    le_device_model = LabelEncoder()
    le_device_model.fit(np.hstack((X[:,1], X_test[:,1])))
    X[:,1] = le_device_model.transform(X[:,1])
    X_test[:,1] = le_device_model.transform(X_test[:,1])
    
    # Standardize features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(np.vstack((X, X_test)))
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    
    return X, y, X_test


#X_train, X_cv, y_train, y_cv, X_test = organize_data()
#num_inputs = X_train.shape[1]
#num_classes = len(set(y_train))
X, y, X_test = organize_data_kf()
num_inputs = X.shape[1]
num_classes = len(set(y))


############# XG Boost model #####################
#pred_prob_cv = xgboost_predict_check(X_train, y_train, X_cv, y_cv, num_round=350)
pred_prob_cv = xgboost_predict_check_kf(X, y, num_inputs, num_classes, kf_size=10, num_round=300)
## Log Loss = 2.38834

############# Neural Net model #########################
#import NN
#hidden_units_1 = 30
#p_dropout = 0.0
#X_train_shared, y_train_shared, X_cv_shared, y_cv_shared, X_test_shared =\
#    NN.load_data_into_shared(X_train, y_train, X_cv, y_cv, X_test = False)
#clf = NN.Network([NN.HiddenLayer(num_inputs, hidden_units_1, p_dropout=p_dropout),\
#                  NN.SoftmaxLayer(hidden_units_1, num_classes, p_dropout=p_dropout)],\
#                 num_batch = 14967, epochs=70, eta=0.4, lmb=0.1)
#clf.fit(X_train_shared, y_train_shared, X_cv_shared, y_cv_shared)
#pred_prob_cv_nn = clf.predict_proba(X_cv_shared)
### Log Loss = 2.42425223454

############# Various sklearn models #############

#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(random_state=0)
### Log Loss = 10.8

#from sklearn.svm import SVC
#clf = SVC(kernel='linear', probability=True)
### Log Loss = 2.40

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier()
### Log Loss = 8.85

#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier()
### Log Loss = 2.48

#pred_prob_cv = sklearn_predict_check(clf, X_train, y_train, X_cv, y_cv)

print "Log Loss = {}".format(logloss(pred_prob_cv, y))