# importing the required module
import numpy as np
from numpy.core.numeric import cross
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# import seaborn as sns
# sns.set_style('whitegrid')
# import statsmodels.api as sm

# https://colab.research.google.com/github/meizmyang/Student-Performance-Classification-Analysis/blob/master/Student%20Performance%20Analysis%20and%20Classification.ipynb#scrollTo=IdD7SgZoHjd9

def prepareForDecisionTree(X : DataFrame):
    X['school'] = [0 if i == 'GP' else 1 for i in X['school']] # GP = 0, MS = 1
    X['sex'] = [0 if i == 'M' else 1 for i in X['sex']] # M = 0, F = 1
    X['address'] = [0 if i == 'U' else 1 for i in X['address']] # U = 0, R = 1
    X['familySize'] = [0 if i == 'LE3' else 1 for i in X['familySize']] # LE3 = 0, GE3 = 1
    X['parentsStatus'] = [0 if i == 'T' else 1 for i in X['parentsStatus']] # T = 0, A = 1
    X['schoolSupport'] = [0 if i == 'no' else 1 for i in X['schoolSupport']] # no = 0, yes = 1
    X['familySupport'] = [0 if i == 'no' else 1 for i in X['familySupport']] # no = 0, yes = 1
    X['paidClasses'] = [0 if i == 'no' else 1 for i in X['paidClasses']] # no = 0, yes = 1
    X['activities'] = [0 if i == 'no' else 1 for i in X['activities']] # no = 0, yes = 1
    X['nursery'] = [0 if i == 'no' else 1 for i in X['nursery']] # no = 0, yes = 1
    X['desireHigherEdu'] = [0 if i == 'no' else 1 for i in X['desireHigherEdu']] # no = 0, yes = 1
    X['internet'] = [0 if i == 'no' else 1 for i in X['internet']] # no = 0, yes = 1
    X['romantic'] = [0 if i == 'no' else 1 for i in X['romantic']] # no = 0, yes = 1

    encoder = OneHotEncoder()
    X = pd.get_dummies(X,columns=['motherJob','fatherJob','reason','guardian'])
    
    # list = []
    # for i in X['motherJob'] : # other = 0, at_home = 1, services = 2, health = 3, teacher = 4
    #     if i == 'at_home':
    #         list.append(1)
    #     elif i == 'services':
    #         list.append(2)
    #     elif i == 'health':
    #         list.append(3)
    #     elif i == 'teacher':
    #         list.append(4)
    #     else:
    #         list.append(0)
    # X['motherJob'] = list

    # list = []
    # for i in X['fatherJob'] : # other = 0, at_home = 1, services = 2, health = 3, teacher = 4
    #     if i == 'at_home':
    #         list.append(1)
    #     elif i == 'services':
    #         list.append(2)
    #     elif i == 'health':
    #         list.append(3)
    #     elif i == 'teacher':
    #         list.append(4)
    #     else:
    #         list.append(0)
    # X['fatherJob'] = list

    # list = []
    # for i in X['reason'] : # other = 0, course = 1, reputation = 2, home = 3
    #     if i == 'course':
    #         list.append(1)
    #     elif i == 'reputation':
    #         list.append(2)
    #     elif i == 'home':
    #         list.append(3)
    #     else:
    #         list.append(0)
    # X['reason'] = list

    # list = []
    # for i in X['guardian'] : # other = 0, father = 1, mother = 2
    #     if i == 'father':
    #         list.append(1)
    #     elif i == 'mother':
    #         list.append(2)
    #     else:
    #         list.append(0)
    # X['guardian'] = list
    return X

def crossValidation(X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234)
    nr_of_splits = 40
    max_depth = 20
    train_error_list_stratified = np.zeros((nr_of_splits,max_depth))
    test_error_list_stratified = np.zeros((nr_of_splits,max_depth))
    for i, (train, test) in enumerate(StratifiedKFold(n_splits=nr_of_splits).split(X,y) ):
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        for j in range (2,max_depth):
            train = []
            test = []
            dtc =  tree.DecisionTreeClassifier(max_depth = j)
            for k in range(10):
                dtc = dtc.fit(X_train, y_train)
                y_train_pred = dtc.predict(X_train)
                y_test_pred = dtc.predict(X_test)
                train_error = 1 - accuracy_score(y_train, y_train_pred)
                test_error = 1 - accuracy_score(y_test, y_test_pred)
                train.append(train_error)
                test.append(test_error)
            train_error_list_stratified[i, j] = np.average(train)
            test_error_list_stratified[i, j] = np.average(test_error) 

        
    plt.title('Stratified classification error, train vs test')
    plt.plot(train_error_list_stratified.mean(0), label = "train")
    plt.plot(test_error_list_stratified.mean(0),label = "test")
    plt.xticks(np.arange(0,max_depth,2))
    plt.xlim([3,max_depth])
    plt.ylabel('Classification error')
    plt.xlabel('Depth')
    plt.legend()
    plt.show()
    

def main():
    mat = pd.read_csv("Dataset/student-combined.csv", sep=';') # load dataset of the math classes 

    mat.columns = ['school','sex','age','address','familySize','parentsStatus','motherEducation','fatherEducation',
            'motherJob','fatherJob','reason','guardian','commuteTime','studyTime','failures','schoolSupport',
            'familySupport','paidClasses','activities','nursery','desireHigherEdu','internet','romantic','familyQuality',
            'freeTime','goOutFriends','workdayAlc','weekendAlc','health','absences','1stPeriod','2ndPeriod','final']

    mat['finalGrade'] = 'None'
    mat.loc[(mat.final >= 13) & (mat.final <= 20), 'finalGrade'] = 1
    mat.loc[(mat.final >= 0) & (mat.final <= 12), 'finalGrade'] = 0

    X = mat.copy(deep=True) # coping mat without changing the original when X is changed

    y = X['finalGrade'].ravel()
    y = y.astype('int')

    X.drop(['1stPeriod','2ndPeriod','final','finalGrade' ], axis=1, inplace=True)

    X = prepareForDecisionTree(X)
    y = np.asarray(y)

    
    attribute_names = X.columns
    class_names = ['fair', 'good']
    X = X.values.tolist() # make X into a list(needed for creating a decision tree)
    X = np.asarray(X)
    
    # crossValidation(X,y)
    dtc = tree.DecisionTreeClassifier(max_depth=17)
    dtc = dtc.fit(X, y)
    plt.figure(figsize=(200,140), dpi=130)
    # plt.figure(figsize=(10,5), dpi=80)

    tree.plot_tree(dtc, feature_names = attribute_names, class_names = class_names, filled =True)
    plt.savefig("decisiontreeNew.png")


    


if __name__ == '__main__':
    main()