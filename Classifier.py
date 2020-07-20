# imports
import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.write("# Datasets")

# chosen dataset and classifier
chosen_ds = st.sidebar.selectbox("Select dataset", ("Iris ", "Breast Cancer", "Digits"))
st.write("The chosen dataset is :", chosen_ds)

chosen_classifier = st.sidebar.selectbox("Select classifier", ("KNN", "SVM", "Random Forest"))


def load_dataset(dataset_name):
    # loading iris as a default dataset
    dataset = datasets.load_iris()
    if dataset_name == "Iris":
        dataset = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        dataset = datasets.load_breast_cancer()
    elif dataset_name == "Digits":
        dataset = datasets.load_digits()
    # loading x and y for the chosen dataset
    X = dataset.data
    Y = dataset.target
    return X, Y

# loading the dataset
x, y = load_dataset(chosen_ds)
# displaying dataset information
st.write('Shape of data', x.shape)
st.write('Number of classes', len(np.unique(y)))

def add_parameter_iu(classifier_name):
    parameters = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        parameters["K"] = K
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        parameters["C"] = C
    else:
        max_tree_depth = st.sidebar.slider('Max depth', 2, 15)
        n_estimators = st.sidebar.slider('number of estimators', 1, 100)
        parameters["max_tree_depth"] = max_tree_depth
        parameters["n_estimators"] = n_estimators
    return parameters

parameters = add_parameter_iu(chosen_classifier)

def create_classifier(classifier_name, parameters):
    if classifier_name == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=parameters["K"])
    elif classifier_name == "SVM":
        classifier = SVC(C=parameters["C"])
    else:
        classifier = RandomForestClassifier(n_estimators=parameters["n_estimators"], max_depth=parameters["max_tree_depth"], random_state=101)

    return classifier

classifier = create_classifier(classifier_name=chosen_classifier, parameters=parameters)

# classification
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=101)

classifier.fit(X_train, Y_train)
y_prediction = classifier.predict(X_test)

accuracy = accuracy_score(Y_test, y_prediction)

st.write(f'the model accuracy is {accuracy}')

# PLOT
pca = PCA(2)
X_projected_data = pca.fit_transform(x)

x1 = X_projected_data[:, 0]
x2 = X_projected_data[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 1")
plt.colorbar()

st.pyplot()
