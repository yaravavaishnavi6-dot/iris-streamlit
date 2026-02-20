import streamlit as st
from sklearn.datasets import load_iris
data=load_iris()

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
x=data.data
y=data.target
model.fit(x,y)

st.header('iris flower classifier')
s1=st.number_input('enter the sepal length')
sw=st.number_input('enter the sepal width')
p1=st.number_input('enter the petal length')
pw=st.number_input('enter the petal width')
y_pred=model.predict([[s1,sw,p1,pw]])
op=data.target_names[y_pred[0]]
st.write(op)
