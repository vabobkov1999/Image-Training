import json
import h5py
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib
import functions as fun



num_trees = 100
test_size = 0.10
seed      = 9
train_path = "dataset/train"
test_path  = "dataset/test"
scoring    = "accuracy"
metrics = ["accuracy", "recall_macro", "precision_macro"]
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    results = {}
    for metric in metrics:
        results[metric] = []

    names = []
    pokemon_names = fun.get_names()
    models = []

    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=seed)))

    h5f_data  = h5py.File('features/data.h5', 'r')
    h5f_label = h5py.File('features/labels.h5', 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string   = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels   = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)
    scoring_relults = []
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

        scoring_relults.append(cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring))

        for metric in metrics:
            cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=metric)
            results[metric].append(cv_results)

        names.append(name)

        #model.fit(trainDataGlobal, trainLabelsGlobal)
        #print(f"[STATUS] {model} report:")
        #print(classification_report(testLabelsGlobal, model.predict(testDataGlobal), target_names=pokemon_names))
    for index in range(len(scoring_relults)):
        msg = "%s: %f (%f)" % (models[index][0], scoring_relults[index].mean(), scoring_relults[index].std())
        print(msg)

    # boxplot algorithm comparison

    for metric in metrics:
        fig = pyplot.figure()
        fig.suptitle(metric)
        ax = fig.add_subplot(111)
        pyplot.boxplot(results[metric])
        ax.set_xticklabels(names)
        pyplot.savefig("boxes/" + metric + ".png")
