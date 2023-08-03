#===========================================================================
#  Author: Kunal SK Sukhija
#===========================================================================

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import joblib

df=pd.read_csv("Car.csv")
# df.drop(columns=("Unnamed: 0"),inplace=True)
print(df.isna().sum())

df=df.dropna()
df=df.reset_index(drop=True)
df=df.drop([840,969])
df=df.reset_index(drop=True)
y=df["selling_price"]
x=df.drop(columns=["Unnamed: 0","torque","selling_price"])
print(x.dtypes)
# print(x.isna().sum())
# print(x.dtypes)
x["max_power"]=x["max_power"].str.replace(" bhp","")
# print(x.isna().sum())

x["engine"]=x["engine"].str.replace(" CC","")
x["mileage"]=x["mileage"].str.replace(" kmpl","")
x["mileage"]=x["mileage"].str.replace(" km/kg","")
x["owner"]=x["owner"].str.replace("Fifth","Fourth & Above Owner")
x["name"]=x["name"].str.split().str[0]

x["max_power"]=pd.to_numeric(x["max_power"])
x["engine"]=pd.to_numeric(x["engine"])
x["mileage"]=pd.to_numeric(x["mileage"])

print(x["name"].value_counts())
print(x.dtypes)
from sklearn.preprocessing import LabelEncoder
lo=LabelEncoder()
lt=LabelEncoder()
x["owner"]=lo.fit_transform(x["owner"])
x["transmission"]=lt.fit_transform(x["transmission"])
joblib.dump(lo,"Joblibs\owner.joblib")
joblib.dump(lt,"Joblibs\Transmission.joblib")

count=x["name"].value_counts()
other=count[count<10]
def f(x):
    if x in other:
        return "Others"
    else:
        return x

x["name"]=x["name"].apply(f)

print(x.isna().sum())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(sparse=False),["seller_type","fuel","name"])],remainder="passthrough")
x=ct.fit_transform(x)
joblib.dump(ct,"Joblibs\onehot.joblib")

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
joblib.dump(sc,"Joblibs\SC.joblib")

from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.2)


# =============================================================================
# Linear Regression
# =============================================================================

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

# =============================================================================
# result=0.82
# =============================================================================

# =============================================================================
# SVM
# =============================================================================

from sklearn.svm import SVR
regressor=SVR(C=1000000,kernel="poly")


# =============================================================================
# SVM with c=50000 and kernel="poly"
# gave result 0.941
# and with c=100000 gave 0.9399
# with c=500000 gave 0.954 and 0.941 and 0.960
# with c=1000000 gave 0.9345
# =============================================================================

regressor.fit(xtrain,ytrain)

ypred=regressor.predict(xtest)

from sklearn.metrics import r2_score
print(r2_score(ytest, ypred))


joblib.dump(regressor,"Joblibs\Regressor.joblib")
# print(df.dtypes)
