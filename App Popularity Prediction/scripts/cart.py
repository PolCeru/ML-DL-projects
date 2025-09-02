#!/usr/bin/python3

from numpy                   import median
from pandas                  import read_csv
from sklearn                 import tree
from sklearn.model_selection import train_test_split

data = read_csv('./google-playstore_preprocessed.csv')

# Split into training and test data
train_data, test_data, train_targets, test_targets = train_test_split(data.drop('Minimum Installs', axis=1), data['Minimum Installs'],test_size=0.2, stratify=data['Minimum Installs'])

score_train = 0
score_test  = 0
trials = 10

score_list_train = []
score_list_test  = []

clf = tree.DecisionTreeClassifier()

for i in range(0, trials):
    clf = clf.fit(train_data, train_targets)
    score_list_train.append(clf.score(train_data, train_targets))
    score_list_test.append(clf.score(test_data, test_targets))
    score_train += clf.score(train_data, train_targets)
    score_test  += clf.score(test_data, test_targets)
    print(i)

print('train_avg='  + str(median(score_list_train)) +
      ' train_med=' + str(score_train/trials) +
      ' test_avg='  + str(median(score_list_test)) +
      ' test_med='  + str(score_test/trials))
