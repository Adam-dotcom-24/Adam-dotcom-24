import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# Simple Advertising Prediction App")
st.write("This app predicts the Sales type!")

data = sns.load_dataset('Tv', 'Newspaper', 'Radio')
X = data.drop(['Sales'],axis=1)
Y = data.species.copy()

modelGaussianIris = GaussianNB()
modelGaussianIris.fit(X, Y)

prediction = modelGaussianIris.predict(df)
prediction_proba = modelGaussianIris.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
