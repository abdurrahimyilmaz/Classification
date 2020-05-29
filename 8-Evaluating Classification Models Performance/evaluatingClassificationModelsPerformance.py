#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:24:12 2020

@author: abdurrahim
"""

"""Confusion matrix
type 1 error = false positive 0ı 1e yuvarlamak yani 0 aslında olan tahminimizde 1 buluyoruz
type 2 error = false negative 1i 0a yuvarlamak yani 1 aslında olan tahminimizde 0 buluyoruz
bunlar daha çok logistic regresyonda sıkıntı özellikle sigmoid fonksiyonunda

cm bize errorları söyler
sol üst 0 0 sağ üstü 0 1 = type 1 error
sol alt 1 0 = type 2 error sağ alt 1 1

accuracy rate = correct / total
error rate = wrong / total
"""

"""accuracy paradox
error tiplerinden kaynaklanır bizim modelimiz outputla çok alakası olan bir parametreyi içermez veya test train verimiz düzgün etiketlenmezse
cm bize sadece error tiplerini söylediğinden biz yüksek accuracy görebiliriz çünkü içerdeki veriyi bilmiyoruz doğru veriyi kullansak belki
accuracymiz düşecek ama yanlış veri ve parametre kullandığımız için yükseldi işte buna paradox deniyor bu da istatistiğin açığı
"""

"""cap curve = cumulative accuracy profile
roc = receiver operating characteristic - roc cap aynı şey değil

perfect - random arası alan a_p
good - random arası alan a_r
1. analiz metodu
AR = a_p / a_r - 1e yaklaştrmak lazım
2. analiz metodu
modelin %50de yeki karşılığı nereye denk geliyor x<%60 rubbish - 60<x<70 poor - good, very good, too good diye gidiyor
"""

"""
In this Part 3 you learned about 7 classification models. Like for Part 2 - Regression, that's quite a lot so you might be asking yourself the same questions as before:

    What are the pros and cons of each model ?

    How do I know which model to choose for my problem ?

    How can I improve each of these models ?

Again, let's answer each of these questions one by one:

1. What are the pros and cons of each model ?

Please find here a cheat-sheet that gives you all the pros and the cons of each classification model.

2. How do I know which model to choose for my problem ?

Same as for regression models, you first need to figure out whether your problem is linear or non linear. You will learn how to do that in Part 10 - Model Selection. Then:

If your problem is linear, you should go for Logistic Regression or SVM.

If your problem is non linear, you should go for K-NN, Naive Bayes, Decision Tree or Random Forest.

Then which one should you choose in each case ? You will learn that in Part 10 - Model Selection with k-Fold Cross Validation.

Then from a business point of view, you would rather use:

- Logistic Regression or Naive Bayes when you want to rank your predictions by their probability. For example if you want to rank your customers from the highest probability that they buy a certain product, to the lowest probability. Eventually that allows you to target your marketing campaigns. And of course for this type of business problem, you should use Logistic Regression if your problem is linear, and Naive Bayes if your problem is non linear.

- SVM when you want to predict to which segment your customers belong to. Segments can be any kind of segments, for example some market segments you identified earlier with clustering.

- Decision Tree when you want to have clear interpretation of your model results,

- Random Forest when you are just looking for high performance with less need for interpretation. 

3. How can I improve each of these models ?

Same answer as in Part 2: 

In Part 10 - Model Selection, you will find the second section dedicated to Parameter Tuning, that will allow you to improve the performance of your models, by tuning them. You probably already noticed that each model is composed of two types of parameters:

    the parameters that are learnt, for example the coefficients in Linear Regression,

    the hyperparameters.

The hyperparameters are the parameters that are not learnt and that are fixed values inside the model equations. For example, the regularization parameter lambda or the penalty parameter C are hyperparameters. So far we used the default value of these hyperparameters, and we haven't searched for their optimal value so that your model reaches even higher performance. Finding their optimal value is exactly what Parameter Tuning is about. So for those of you already interested in improving your model performance and doing some parameter tuning, feel free to jump directly to Part 10 - Model Selection.
"""