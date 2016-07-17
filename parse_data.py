# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 23:14:40 2016

Obviously needs to do better than 

@author: Tony
"""

#import numpy as np
#import matplotlib.pylab as plt
import pandas as pd
import pickle

gender_age_train = pd.read_csv('gender_age_train.csv')
# gender_age_test = pd.read_csv('gender_age_test.csv')
app_events = pd.read_csv('app_events.csv')
app_labels = pd.read_csv('app_labels.csv')
events = pd.read_csv('events.csv')
label_categories = pd.read_csv('label_categories.csv')
phone_brand_device_model = pd.read_csv('phone_brand_device_model.csv')

events.join()

#Translate names in phone brands
phone_brand_device_model['phone_brand'] = phone_brand_device_model

with open('combined_df.pkl', 'w'):
    pickle.dump(events, f)