# imports
import streamlit as st
from sklearn import datasets
import numpy as np

st.write("# Datasets")

# chosen dataset and classifier
chosen_ds = st.sidebar.selectbox("Select dataset", ("Iris ", "Breast Cancer", "Digits"))
st.write("The chosen dataset is :", chosen_ds)

chosen_classifier = st.sidebar.selectbox("Select classifier", ("KNN", "SVM", "Random Forest"))


def load_dataset(dataset_name):
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
