import sys
import pandas as pd
from preprocessing_one import read_csv_dataset
from helper_functions import count_hashtags
from sklearn.metrics import recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def process_data(csv_filename, rawtweets_filename):
    data = read_csv_dataset(csv_filename)

    # Get the class attribute as our label
    labels = data['class']

    data['num_hashtags'] = pd.Series(count_hashtags(rawtweets_filename))


    # Drop the labels from the other dataset
    data.drop('class', axis=1, inplace=True)

    # Also drop the id, as it isn't a useful feature
    data.drop('id', axis=1, inplace=True)

    # Have a value of 1 for the label if there is an ADR
    # 0 otherwise
    y = pd.Series([0,1], index=["N", "Y"])
    labels = labels.map(y)

    features = data
    
    return features, labels

def run_cross_validation(param_dict, clf_in, features, labels):
    

    clf = GridSearchCV(clf_in, param_dict, scoring = make_scorer(f1_score))

    grid_fit = clf.fit(features, labels)

    print "best classifier is:"

    print grid_fit.best_estimator_

    return grid_fit.best_estimator_

def run_pca(num_components, features):

    pca = PCA(n_components=num_components)

    pca.fit(features)

    reduced_data = pca.transform(features)

    reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension ' + str(a) for a in range(1,num_components+1)])
    
    return reduced_data
    
didPCA = False
PCA_comps = 10


features, labels = process_data("train_frequencies.csv", "train.txt")

if len(sys.argv) is 1:
    print "No classifier specified - Naive Bayes chosen by default"
    clf = MultinomialNB()
else:
    if len(sys.argv) > 2:
        if sys.argv[2] == 'y':
            print "PCA is being run"
            didPCA = True
            features = run_pca(PCA_comps, features)
        else:
            print "PCA not chosen"
        
    
    if sys.argv[1] == 'dt':
        param_dict = parameters_dt = {'criterion' : ['gini', 'entropy'], 'max_depth' : [2,5,10,50], 'min_samples_split' : [2,5,10,25,100]}
        clf = run_cross_validation(param_dict, DecisionTreeClassifier(), features, labels )
    elif sys.argv[1] == 'svm':
        param_dict = {'kernel' : ['linear', 'poly', 'rbf'], 'C' : [1, 10, 100, 200] }
        clf = run_cross_validation(param_dict, SVC(), features, labels)
    elif sys.argv[1] == 'rf':
        param_dict = {'criterion' : ['gini', 'entropy'], 'max_depth' : [2,5,10,50], 'min_samples_split' : [2,5,10,25,100]}
        clf = run_cross_validation(param_dict, RandomForestClassifier(), features, labels)
    elif sys.argv[1] == 'nb':
        clf = MultinomialNB()
    else:
        print "Incorrect format of classifer has been input.\ndt - Decision Tree Classifier\nsvm - Support Vector Classifier\nrf - Random Forest Classifier\nNaive Bayes chosen by default"
        clf = MultinomialNB()
        

test_features, test_labels = process_data("test_frequencies.csv", "test.txt")

if didPCA:
    test_features = run_pca(PCA_comps, test_features)


clf.fit(features, labels)

pred = clf.predict(test_features)

print "F1:"

print f1_score(test_labels, pred)


print "Precision:"

print precision_score(test_labels, pred)


print "Recall:"

print recall_score(test_labels, pred)











