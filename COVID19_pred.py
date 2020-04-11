#importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
from datetime import timedelta
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR


#importing the dataset
covid = pd.read_csv("covid_19_data.csv")
covid.head()

print("Size/Shape of Data :" , covid.shape)
print("Check For Null Values:\n ", covid.isnull().sum())
print("Checking Data-Type: ", covid.dtypes)

#droping the serial number column (SNO)
covid.drop(["SNo"] , 1, inplace = True)


covid["ObservationDate"] = pd.to_datetime(covid["ObservationDate"])


#Basic INFORMATION
print("Basic Information")
print("Total number of confirmed cases around the wrold", datewise["Confirmed"].iloc[-1])
print("Total number of recovered cases around the wrold", datewise["Recovered"].iloc[-1])
print("Total number of deaths cases around the wrold", datewise["Deaths"].iloc[-1])
print("Total number of Active cases around the wrold", datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1])
print("Total number of Closed cases around the wrold", datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1])

#Ploting graph of active cases
plt.figure(figsize=(15,5))
sns.barplot(x=datewise.index.date,y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])
plt.title("Distribution plot for active cases")
plt.xticks(rotation=90)

#Plotting graph of closed cases
plt.figure(figsize=(15,5))
sns.barplot(x=datewise.index.date,y=datewise["Recovered"]+datewise["Deaths"])
plt.title("Distribution plot for Closed cases")
plt.xticks(rotation=90)


datewise["Days Since"] = datewise.index-datewise.index[0]
datewise["Days Since"] = datewise["Days Since"].dt.days
train_ml = datewise.iloc[:int(datewise.shape[0]*0.90)]
valid_ml = datewise.iloc[int(datewise.shape[0]*0.90):]
model_scores = []

lin_reg = LinearRegression(normalize = True)
svm = SVR(C=1, degree=6, kernel="poly" , epsilon=0.01)
lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))


new_date = []
new_prediction_lr = []
new_prediction_svm = []
for i in range(1,18):
    new_date.append(datewise.index[-1]+timedelta(days=i))
    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])
    new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
pd.set_option("display.float_format",lambda x:'%.f' %x)
model_predictions = pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_svm),columns = ["Dates","LINEAR REGRSN","SVM PREDICTION"])
model_predictions.head(10)
