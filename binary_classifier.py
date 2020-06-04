from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import csv
import pandas as pd
import numpy as np
import pickle
import importlib,sys
importlib.reload(sys)
csv_data = pd.read_csv('./train.csv')  # read train.csv
data = np.array(csv_data)  #transform it into numpy format

x = data[0:160000,1:37]  #get the features
y = data[0:160000,37:38] #get the tags
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.90,test_size=0.10)
#scrambling for training, and using 90% of training dataset as training, 10% as testing dataset and calculating prediction accuracy
#because the test dataset itself cannot be used for prediction without tag

##We can choose the following classifiers. After testing, I found that the performance of randomforest is the best
##and we can further tune parameters to get better performance

#classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo') #svm
classifier =RandomForestClassifier()#随机森林
#classifier=KNeighborsClassifier()#K邻
#classifier=tree.DecisionTreeClassifier()#决策树
#classifier=GradientBoostingClassifier()#梯度增强决策树
#classifier= AdaBoostClassifier()#adaboost

classifier.fit(train_data, train_label.ravel())
from sklearn.metrics import accuracy_score
tra_label = classifier.predict(train_data)
tes_label = classifier.predict(test_data)
print("训练集：", accuracy_score(train_label, tra_label))  #train_accurate
print("测试集：", accuracy_score(test_label, tes_label))  #test_accurate

# read"test.csv"and transform it into numpy format
resultData = pd.read_csv('./test.csv')
result_Q = np.array(resultData)[:, 1:37]

import importlib, sys
importlib.reload(sys)
# get output
output = classifier.predict(result_Q)
print('predict_result:\n', output)

f = open('result.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)
# write result
for i in range(40000):
    csv_writer.writerow([str(i + 160001), str(output[i])])

f.close()




