import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
#Data Preprocessing
data = pd.read_csv("Coffe_sales.csv")
data["date"] = pd.to_datetime(data["date"] , errors='coerce')
data["datetime"] = pd.to_datetime(data["datetime"] , errors='coerce')
data["card"] = data["card"].fillna(0)
data["date"].fillna(method = "ffill" , inplace = True)
data["datetime"].fillna(method = "ffill" , inplace = True)
data.drop(columns = ["Monthsort" , "Weekdaysort" , "card"] , inplace = True)
le = LabelEncoder()
data["en_cash_type"] = le.fit_transform(data["cash_type"])
data["en_Time_of_Day"] = le.fit_transform(data["Time_of_Day"])
data["en_Weekday"] = le.fit_transform(data["Weekday"])
data["en_Month_name"] = le.fit_transform(data["Month_name"])
data["en_coffee_name"] = le.fit_transform(data["coffee_name"])
data["money"] = data["money"].str.replace('R' , '').astype(float , copy = True)
data.drop(columns = ["cash_type" , "Weekday" , "Time_of_Day" , "datetime"] , inplace = True)
print(data.info())
print(data.shape)
#Model for coffe_name corresponding to sales(money)
plt.figure(figsize=(10,6))
sns.countplot(data = data , x = "en_coffee_name" , order = data["en_coffee_name"].value_counts().index)
plt.show()
x_train , x_test , y_train , y_test = train_test_split(data[["en_coffee_name"]] , data["money"] , test_size=0.2 , random_state=42)
rfr = RandomForestRegressor(n_estimators=100 , random_state=42)
rfr.fit(x_train,y_train)
y_pred = rfr.predict(x_test)
#model accuracy 85%

#graphical representation for daily sales
ds = data.groupby("date")["money"].sum().reset_index()
plt.figure(figsize=(10,6))
sns.barplot(x = "date" , y = "money" , data = ds)
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
print(ds)

#Model to predict sales as per month
dt = data.groupby(["Month_name" , "en_Month_name"])["money"].sum().reset_index()
plt.figure(figsize = (10,6))
sns.barplot(x = "Month_name" , y = "money" , data = dt)
plt.title("Month vs sales")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
print(dt)
# #1 --> train 2 --> test
x_1 , x_2 , y_1 , y_2 = train_test_split(dt[["en_Month_name"]] , dt["money"] , test_size=0.2 , random_state=42)
lr = LinearRegression()
lr.fit(x_1 , y_1)
y_pred1 = lr.predict(x_2)
print(r2_score(y_2 , y_pred1)*100)#45% Accuracy

#Model to predict which time of the day has highest sales (general for overall data)
da = data.groupby(["date" , "en_Time_of_Day"])["money"].sum().reset_index()
plt.figure(figsize=(10,6))
sns.barplot(x = "en_Time_of_Day" , y = "money" , data = da)
plt.title("Time of day V/S Sales")
plt.xlabel("Time of day")
plt.ylabel("Sales")
plt.show()
x_train1 , x_test1 , y_train1 , y_test1 = train_test_split(da[["en_Time_of_Day"]] , da["money"] , test_size=0.2 , random_state=42)
rfr.fit(x_train1,y_train1)
y_pred2 = rfr.predict(x_test1)
print(da)
