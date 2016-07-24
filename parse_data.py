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

gender_age_train = pd.read_csv('gender_age_train.csv', index_col=0)
# gender_age_test = pd.read_csv('gender_age_test.csv')
app_events = pd.read_csv('app_events.csv')
app_labels = pd.read_csv('app_labels.csv', index_col=0)
events = pd.read_csv('events.csv', index_col=0)
label_categories = pd.read_csv('label_categories.csv', index_col=0)
dev_df = pd.read_csv('phone_brand_device_model.csv')

events = events.join(app_events, sort=False)
# This next line needs further exploration
#events = events.merge(app_labels, left_index='app_id', right_index=True)

#Translate names in phone brands
#transl_table = pd.read_csv('name_translation.txt', delimiter=' ', encoding='utf-8')

####### Create new features #######

#List the number of events on the device
counted_events = events.drop(['timestamp','longitude','latitude'],axis=1)
counted_events['num_events'] = 0
counted_events = counted_events.groupby(['device_id'],sort=False).count()
dev_df = dev_df.join(counted_events, sort=False)
dev_df['num_events'] = dev_df['num_events'].fillna(0)

#List the number of 

#Save data
with open('dev_df.pkl', 'w') as f:
    cPickle.dump(dev_df, f)
    
with open('events.pkl', 'w') as f:
    cPickle.dump(events, f)