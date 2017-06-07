# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    cross_validation = False
    random_forest = False

    data = pd.read_csv('..\\Data\\car.data', header=None)
    n_columns = len(data.columns)
    columns = ['buy', 'maintain', 'doors', 'persons', 'boot', 'safety', 'accept']
    new_columns = dict(zip(np.arange(n_columns), columns))
    data.rename(columns=new_columns, inplace=True)
    for col in columns:
        data[col] = pd.Categorical(data[col]).codes
    print data
    x = data.loc[:, columns[:-1]]
    y = data['accept']
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.7)
    if random_forest:
        clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=12, min_samples_split=5, max_features=5)
    else:
        clf = DecisionTreeClassifier(criterion='gini', max_depth=12, min_samples_split=5, max_features=5)
    if cross_validation:
        model = GridSearchCV(clf, param_grid={'max_depth': np.arange(10,20),
                                              'min_samples_split': np.arange(5, 20),
                                              'max_features': np.arange(1, 7)
                                              }, cv=3)
        model.fit(x, y)
        print model.best_params_
        clf = model.best_estimator_
    else:
        clf.fit(x, y)
    y_hat = clf.predict(x)
    print '训练集精确度：', metrics.accuracy_score(y, y_hat)
    y_test_hat = clf.predict(x_test)
    print '测试集精确度：', metrics.accuracy_score(y_test, y_test_hat)
    n_class = len(data['accept'].unique())
    y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
    y_test_one_hot_hat = clf.predict_proba(x_test)
    fpr, tpr, _ = metrics.roc_curve(y_test_one_hot.ravel(), y_test_one_hot_hat.ravel())
    print 'Micro AUC:\t', metrics.auc(fpr, tpr)
    print 'Micro AUC(System):\t', metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat, average='micro')
    auc = metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat, average='macro')
    print 'Macro AUC:\t', auc

    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 7), dpi=80, facecolor='w')
    plt.plot(fpr, tpr, 'r-', lw=2, label='AUC=%.4f' % auc)
    plt.legend(loc='lower right')
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.grid(b=True, ls=':')
    plt.title(u'ROC曲线和AUC', fontsize=18)
    plt.show()
