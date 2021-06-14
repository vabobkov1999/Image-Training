import h5py
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import functions as fun

num_trees = 100
test_size = 0.10
seed      = 9
train_path = "dataset/train"
test_path  = "dataset/test"

if __name__ != '__main__':
    pokemon_names = fun.get_names()
    h5f_data  = h5py.File('features/data.h5', 'r')
    h5f_label = h5py.File('features/labels.h5', 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string   = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels   = np.array(global_labels_string)
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

    clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)

    # loop through the test images
    for file in glob.glob('data/test' + "/*.*"):
        print(file)
        image = cv2.imread(file)
        image = cv2.resize(image, (750,750))
        fv_hu_moments = fun.fd_hu_moments(image)
        fv_haralick   = fun.fd_haralick(image)
        fv_histogram  = fun.fd_histogram(image)

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        prediction = clf.predict(global_feature.reshape(1,-1))[0]

        image2 = cv2.putText(image, pokemon_names[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        cv2.imwrite( 'data/result/' + file.split('/')[2], image2)
