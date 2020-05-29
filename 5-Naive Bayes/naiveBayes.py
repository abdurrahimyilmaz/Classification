#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:04:18 2020

@author: abdurrahim
"""
"""
bayes theorem = b verilmişken anın olma olasılığı = a verilmişken bnin olma olasılığı * a sayısı / b sayısı
posterior probability = likelihood * prior probability / marginal likelihood
bayes teoreminden yeni bi nokta çıktığında olasılık hesaplayak ne tarafa yakın olduğunu olasılığı yüksek olanı seçerek buluruz
daha doğrusu birisini bulsak yeterli
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values #matrix
y = dataset.iloc[:, 4].values #vector

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling - lr otomatik yapmıyor manuel yapmamız gerekiyor
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#fitting classifier to training test
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() # argüman yok bu kadar basit aslında
classifier.fit(x_train,y_train)

#predict test set results
y_pred = classifier.predict(x_test)

#making the confusion matrix = biz modelin predictive powerını görmüş olduk
from sklearn.metrics import confusion_matrix #sol üst sağ alt doğru tahminler sağ üst sol alt yanlış tahminler
cm = confusion_matrix(y_test, y_pred)

#visualising test= gerçek sonuçları ve tahmin bölgelerini görmemizi sağlar
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#visualising train= gerçek sonuçları ve tahmin bölgelerini görmemizi sağlar
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('classifier (Train set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()