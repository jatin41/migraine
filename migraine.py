import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

migraine_dataset = pd.read_csv('/Users/jatinagrawal/__pycache__/Migraine.csv')
migraine_dataset.head()
migraine_dataset.tail()
migraine_dataset.shape
migraine_dataset['Type'].value_counts()
migraine_dataset.groupby('Type').mean()
X=migraine_dataset.drop(columns='Type', axis=1)
Y=migraine_dataset['Type']
print(X)
print(Y)

scaler = StandardScaler()
scaler.fit(X,y=None,sample_weight=None)
standardized_data = scaler.transform(X)
print(standardized_data)
X = standardized_data
Y = migraine_dataset['Type']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape , X_test.shape)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
X_train_prediciton = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediciton, Y_train)
print('Accuracy on training data: ', training_data_accuracy)
X_test_prediciton = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediciton, Y_test)
print('Accuracy on test data: ', test_data_accuracy)

Age = st.number_input(' Age ')
Duration = st.number_input(' Duration ')
frequency = st.number_input(' frequency ')
Location = st.number_input(' Location ')
character = st.number_input(' character ')
intensity = st.number_input(' intensity ')
Nausea = st.number_input(' Nausea ')
Vomit = st.number_input(' Vomit ')
Phonophobia = st.number_input(' Phonophobia ')
photophobia = st.number_input(' photophobia ')
visual = st.number_input(' visual ')
sensory = st.number_input(' sensory ')
Dysphasia = st.number_input(' Dysphasia')
Dysarthria = st.number_input(' Dysarthria ')
vertigo=st.number_input(' Vertigo ')
Tinnitus = st.number_input(' Tinnitus ')
Hypoacusis = st.number_input(' Hypoacusis ')
Diplopia = st.number_input(' Diplopia ')
Defect=st.number_input(' Defect ')
Ataxia = st.number_input(' Ataxia ')
Conscience = st.number_input(' Conscience ')
parathesia = st.number_input(' parathesia ')
DPF = st.number_input(' DPF ')

input_data = (Age,Duration,frequency,Location,character,intensity,Nausea,Vomit,Phonophobia,photophobia,visual,sensory,Dysphasia,Dysarthria,vertigo,Tinnitus,Hypoacusis,Diplopia,Defect,Ataxia,Conscience,parathesia,DPF)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
if st.button("predict"):
    st.write(prediction)
print(prediction)


