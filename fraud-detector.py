import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('/Users/denizgulal/Desktop/fraud.csv')

#print(data)

#Visualize the types of payments

type = data['type'].value_counts()
transaction = type.index
quantity = type.values

explode = [0.1, 0.2, 0.3, 0.2, 0.2]
pie_diagram = plt.pie(quantity, labels=transaction, autopct='%1.1f', explode=explode )

plt.show()

#Check the correlation with the other features of the data and fraud

correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))

#Arrange data to have a better understanding
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())

#Splitting the data for test and training
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

#Training the ml model
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

#Predict a new transaction 
features = np.array([[2, 9000.60, 9000.60, 1500.0]])
print(model.predict(features))

