# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import cross_validation, neighbors
from matplotlib import pyplot as plt
                
data_frame = pd.read_csv("<location to the data set (.csv file)>");                 
train = np.array(data_frame.drop(['late'], 1));
test = np.array(data_frame['late']);
                
x_train, x_test, y_train, y_test = cross_validation.train_test_split(train, test, test_size = 0.4);

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train);

accuracy = clf.score(x_test, y_test);
print('Accuracy: {}'.format(accuracy));

#Predicting Test Data
pre_data = np.array([[15, 8.42, 707, 803, 813, 2, 815, 933, 945]]); #test data for prediction...
pre_data = pre_data.reshape(len(pre_data),-1);
prediction = clf.predict(pre_data);

#Prediction....
print(prediction);
print(type(data_frame['late']));     
data_frame['late'].plot();
plt.plot(pre_data[0], color = 'red');
