#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    path = '..\\Regression\\iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x = data[range(4)]
    y = pd.Categorical(data[4]).codes
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=0)
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=6, min_samples_leaf=3)
    clf.fit(x, y)
    fis = clf.feature_importances_
    iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
    print 'feature importances:', fis
    for f_name, fi in zip(iris_feature, fis):
        print '%s:%.6f' % (f_name, fi)
    y_pred = clf.predict(x)
    print 'train accuracy:', accuracy_score(y, y_pred)
    y_test_pred = clf.predict(x_test)
    print 'test accuracy:', accuracy_score(y_test, y_test_pred)
