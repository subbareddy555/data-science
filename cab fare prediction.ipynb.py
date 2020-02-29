#!/usr/bin/env python
# coding: utf-8

# # Project Name – Cab Fare Prediction
# 
# 

# # Problem Statement -

# You are a cab rental start-up company. You have successfully run the pilot project and
# now want to launch your cab service across the country. You have collected the
# historical data from your pilot project and now have a requirement to apply analytics for
# fare prediction. You need to design a system that predicts the fare amount for a cab ride
# in the city.

# # Importing the required libraries for the project

# In[1]:


#installing packages:!pip install [package Name]
#Importing required libraries
import os #getting access to input files
import pandas as pd # Importing pandas to perform EDA(Exploratory Data Analysis)
import numpy as np  # Importing numpy to perform Linear Algebric operations
import matplotlib.pyplot as plt # Importing for Data Visualization
import seaborn as sns # Importing for Data Visualization
from collections import Counter #Import this to check on the data and count
from sklearn.tree import DecisionTreeRegressor # import ML algorithm to train a model
from sklearn.ensemble import RandomForestRegressor #import ML algorithm to train a model
from sklearn.ensemble import GradientBoostingRegressor #ML algorithm to train a model
from sklearn.linear_model import LinearRegression #ML algorithm to train a model
from sklearn.model_selection import train_test_split #importing this library to split the dataset into train and test dataset
from sklearn.metrics import mean_squared_error # importing this to check the residual error between actual and predicted values
from sklearn.metrics import r2_score #importing this library to find accuracy for regression model
from pprint import pprint
from sklearn.model_selection import GridSearchCV #Importing this library for tuning the hyper parameters   


# # Importing the datasets into dataframes 

# In[2]:


train=pd.read_csv("C:/Users/User/Desktop/Cab Fare Prediction Project/train_cab.csv")
test=pd.read_csv("C:/Users/User/Desktop/Cab Fare Prediction Project/test.csv")


# # Checking for the null data values in each data frame individually

# In[3]:


train.isnull().sum()


# In[4]:


# Check for missing values present in whole training datset.

print("\n The missing value percentage in training data :")

#Create dataframe with missing percentage
missing_val = pd.DataFrame(train.isnull().sum())

#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_values'})

#Calculate new variable missing value percentage
missing_val['Missing_Value_Percentage'] = (missing_val.Missing_values/len(train))*100

#descending order
missing_val = missing_val.sort_values('Missing_Value_Percentage', ascending=False).reset_index(drop=True)
missing_val


# In[5]:


train.dtypes


# In[6]:


train.shape


# # Observation:
# From the above code we can observe that the null values available in the given dataset is less than 10% and hence can be dropped with out any issues, also we have to note that the object datatypes of the columns "fare_amount" and "pickup_datetime" needs to be converted.

# In[7]:


train.head()


# In[8]:


train.tail()


# # Exploring the test dataset(test)
# the training dataset have some null values, we will do Exploratory data analysis on this moving forward, we consider the test data set(test)

# In[9]:


test.isnull().sum()


# In[10]:


test.dtypes


# In[11]:


test.shape


# In[12]:


test.head()


# In[13]:


test.tail()


# # Observation:
# From the above code we can come to a conclusion that the testing dataset(test) had no null values and hence there is no need to drop any values from this dataset, also note that all the datatypes available in test dataset that is test doesn't need any conersion

# # Handling the missing values on training dataset(train)
# 
# Handling the missing values include all the columns of dataframe(train)
# First of all we have to convert the object datatype of the column "fare_amount" to float datatype 

# In[14]:


train["fare_amount"]=pd.to_numeric(train["fare_amount"],errors = "coerce")


# In[15]:


#Rechecking for the conversion of fare_amount
train.dtypes


# In[16]:


#dropping all the null values from the training dataset on the specific column of "pickup_datetime"
train.dropna(subset= ["pickup_datetime"])


# In[17]:


#Confirming that no null values available on the "pickup_datetime" column
train.isnull().sum()


# In[18]:


#converting all the values available  timestamp of "pickup_datetime" and picking up some important values from it
train['pickup_datetime'] =  pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC',errors="coerce")
#the coerce option in the above will ensure that non-numeric values be converted into NaN


# In[19]:


# Now we have spliting the datatime so we need to Adding the columns to the existing dataframe (train)
train['year'] =train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minute'] = train['pickup_datetime'].dt.minute


# In[20]:


#Confirming that new columns were added to the dataframe(train)
train.dtypes


# In[21]:


#Working on pickup_datetime column in order to extract important values from it 
#converting all the values available  timestamp of "pickup_datetime" and picking up some important values from it
test['pickup_datetime'] =  pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC',errors="coerce")


# In[22]:


#Extracting and assigning the important values to the existing dataframe(df1)
test['year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minute'] = test['pickup_datetime'].dt.minute


# In[23]:


#reconfirming that new columns added to test data set(test)
test.dtypes


# In[24]:


train.head()


# In[25]:


test.head()


# In[26]:


train.shape


# In[27]:


test.shape


# # Exploration of datasets
# 
# This step includes all the statistical values such as central tendency for the available dataframes which gives us more details such as minimum and maximum values of each column preseent in the dataset

# In[28]:


train.describe(include="all")


# In[29]:


#checking for null values
train["pickup_datetime"].isnull().sum()


# In[30]:


#dropping the null values on "pickup_datetime"
train=train.drop(train[train["pickup_datetime"].isnull()].index,axis=0)


# In[31]:


#now we have to check it still any null values in our Pickup_datetime
train["pickup_datetime"].isnull().sum()


# In[32]:


#working on the passenger_count column
train["passenger_count"].describe(include="all")


# # Observation:
# From the above code, we can observe that the maximum number of passengers for a ride is 5345 which is not at all possible and we can conclude the values greater than 6(maximum passengers count for a car like SUV) & the count with 0 passengers will be dropped as outliers
# 

# # Working on column "passenger_count"

# In[33]:


train = train.drop(train[train["passenger_count"]> 6 ].index, axis=0)
train = train.drop(train[train["passenger_count"]==0 ].index, axis=0)


# In[34]:


train.shape


# In[35]:


train["passenger_count"].sort_values(ascending=True)


# In[36]:


#now drop the values which have values is NULL and passenger count is float values is also drop
train = train.drop(train[train["passenger_count"].isnull() ].index, axis=0)
train = train.drop(train[train["passenger_count"]==0.12 ].index, axis=0)


# In[37]:


train.shape


# # Working on the column "fare_amount"
# 
# The outlier data of this particular column "fare_amount" can be seen as 54343.0 and standard deviation roughly as 432. Hence we will drop the values greater than 500 and the values less than 1(after checking that the dropping of these values isn't dropping the size of dataset drastically)

# In[38]:


train["fare_amount"].sort_values(ascending=False)


# In[39]:


train["fare_amount"].describe()


# In[40]:


train = train.drop(train[train["fare_amount"]>500  ].index, axis=0)
train = train.drop(train[train["fare_amount"]<1 ].index, axis=0)
train = train.drop(train[train["fare_amount"].isnull()].index, axis=0)


# In[41]:


train.shape


# In[42]:


train.head()


# In[43]:


train.tail()


# # Working on the columns "pickup_latitude","pickup_longitude","dropoff_latitude" & "dropoff_longitude"
# 
# we know the range of latitude is -90 to 90
# and also the range of longitude is -180 to 180, hence we are dropping all the values of latitude and longitude which were out of range.

# In[44]:


train[train["pickup_latitude"]>90]
#train["pickup_latitude"].shape


# In[45]:


train[train["pickup_latitude"]<-90]


# In[46]:


train[train["pickup_longitude"]<-180]


# In[47]:


train[train["pickup_longitude"]>180]


# In[48]:


train[train["dropoff_latitude"]>90]


# In[49]:


train[train["dropoff_latitude"]<-90]


# In[50]:


train[train["dropoff_longitude"]<-180]


# In[51]:


train[train["dropoff_longitude"]>180]


# In[52]:


train = train.drop(train[train["pickup_latitude"]>90].index, axis=0)
train = train.drop(train[train["pickup_latitude"]<-90].index, axis=0)
train = train.drop(train[train["pickup_longitude"]>180].index, axis=0)
train = train.drop(train[train["pickup_longitude"]<-180].index,axis=0)


# In[53]:


train = train.drop(train[train["dropoff_latitude"]>90].index, axis=0)
train = train.drop(train[train["dropoff_latitude"]<-90].index, axis=0)
train = train.drop(train[train["dropoff_longitude"]>180].index, axis=0)
train = train.drop(train[train["dropoff_longitude"]<-180].index,axis=0)


# In[54]:


train.shape


# In[55]:


train.isnull().sum()


# In[56]:


test.isnull().sum()


# # Calculating the distance based on latitude and longitude:
#     we are having the values of latitude and longitude, hence we can calculate the distance travelled by a passenger so that we can have only one input feature instead of four.
# This helps in reduction of dimensions of input features which helps improving the model accuracy. We will calculate the distance using haversine formula.
#     

# In[57]:


#As we know that we have given pickup longitute and latitude values and same for drop. 
#So we need to calculate the distance Using the haversine formula and we will create a new variable called distance
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km


# # Applying the haversine formula on  both train and test datasets

# In[58]:


train['distance'] = train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[59]:


test['distance'] = test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[60]:


train.head()


# Eliminating the outliers in the new column "distance"

# In[61]:


train["distance"].sort_values(ascending=False)


# In[62]:


train["distance"].describe(include="all")


# In[63]:


train["distance"].sort_values(ascending=False).head(40)


# In[64]:


train=train.drop(train[train["distance"]>130].index,axis=0)
train=train.drop(train[train["distance"]==0].index,axis=0)


# In[65]:


train.shape


# In[66]:


train.head()


# After the datapreprocessing and data cleansing dropping the unnecessary columns present in train and test datasets.
# This is indeed known as "Data preparations for the models"

# In[67]:


cols=["pickup_datetime","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","Minute"]
train=train.drop(cols,axis=1)


# In[68]:


train.head()


# Converting the all the datatypes of "columns" into desired format

# In[69]:


train["passenger_count"]=train["passenger_count"].astype("int64")
train["year"]=train["year"].astype("int64")
train["Month"]=train["year"].astype("int64")
train["Date"]=train["Date"].astype("int64")
train["Day"]=train["Day"].astype("int64")
train["Hour"]=train["Hour"].astype("int64")


# In[70]:


train.head()


# In[71]:


train.dtypes


# In[72]:


test=test.drop(cols,axis=1)


# In[73]:


test.head()


# In[74]:


test.dtypes


# # Visualization of Data

# Drawing the plot of passengers travelled in the cab

# In[75]:


#passenger count visualization
plt.figure(figsize=(7,7))
sns.countplot(x="passenger_count", data=train)


# # Observation:
#     From the above graph we can observe that the most of the rides were availed by one or two passengers at a time.

# Drawing the plot for "passenger_count" vs "fare_amount"

# In[76]:


#Relationship beetween number of passengers and Fare

plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=20) #s means here ,s=number of dots
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.title("Passenger_count Vs Fare_amount")
plt.show()


# # Observation:
# 
# From the above graph we can observe that the revenue is high from the rides availed by one or two passengers at a time

# Drawing the plot for "Date" vs "fare_amount"

# In[77]:


#Relationship between date and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'], y=train['fare_amount'], s=20)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.title("Date Vs Fare_amount")
plt.show()


# # Observation:
# From the above graph we can see that highest fare was charged on 3rd and 24th of the month
#     

# Plotting the graph for number of raids per hour in a day

# In[78]:


#graph for hourly cab bookings
plt.figure(figsize=(8,7))
train.groupby(train["Hour"])['Hour'].count().plot(kind="bar")
plt.title("Hourly Cab Bookings")
plt.show()


# Observation:
#     We can confirm that least number of rides were at 5AM and more rides most number of rides were taken at 6PM and 7PM, hence the more number of cars can be arranged at those rush hours.

# Plotting graph for "Hour" vs "fare_amount"

# In[79]:


#Relationship between Time and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'], y=train['fare_amount'], s=10)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.title("Time Vs Fare_amount")
plt.show()


# Observation:
#     The highest fare was 8AM in the morning and 10PM in the night of a day.

# Graph for number of rides per day

# In[80]:


#impact of Day on the number of cab rides
plt.figure(figsize=(8,7))
sns.countplot(x="Day", data=train)


# Observation: we can see that there is no much difference or variance among the days of the week, the day is not impacting much on the number of rides

# Graph for "Day" vs "fare_amount"

# In[81]:


#Relationships between day and Fare
plt.figure(figsize=(8,7))
plt.scatter(x=train['Day'], y=train['fare_amount'], s=15)
plt.xlabel('Day')
plt.ylabel('Fare')
plt.title("Day Vs Fare_amount")
plt.show()


# Observation: The highest fare was charged on day0 and day3 i.e., Sunday and Wednesday.

# Graph between "Distance" vs "fare_amount"

# In[82]:


#Relationship between distance and fare 
plt.figure(figsize=(8,7))
plt.scatter(x = train['distance'],y = train['fare_amount'],c = "r")
plt.xlabel('Distance')
plt.ylabel('Fare')
plt.title("Distance Vs Fare_amount")
plt.show()


# Observations: Most number of rides were taken in between the distance between 0 to 40kms, also the highest fare amount been charged with in this limit.

# # Plotting the distribution of the columns "fare_amount" & "distance" in train dataset

# In[83]:


#Normality check of training data is uniformly distributed or not-

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='red')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# Observations: the distribution of "fare_amount" & "distance" were right skewed in order to get right predictions we will transform the values of these two columns using logarithmic function.
#     

# # Transforming the values of columns "fare_amount" & "distance" on train dataset

# In[84]:


#since skewness of target variable is high, apply log transform to reduce the skewness-
train['fare_amount'] = np.log1p(train['fare_amount'])

#since skewness of distance variable is high, apply log transform to reduce the skewness-
train['distance'] = np.log1p(train['distance'])


# Plotting the distributions of "fare_amount" & "distance" after transformation on train dataset

# In[85]:


#Normality Re-check to check data is uniformly distributed or not after log transformartion

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='red')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# Observations: the distributions of "fare_amount" & "distance" are not skewed and hence they are ready for the training of a model.

# Checking the distribution of "distaance" on the test dataset

# In[86]:


#Normality check for test data is uniformly distributed or not-

sns.distplot(test['distance'],bins='auto',color='red')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# Observation: The distance in test dataset is skewed and needs a transformation before predicting the target label.

# # Transforming the "distance" column on test dataset

# In[87]:


#since skewness of distance variable is high, apply log transform to reduce the skewness-
test['distance'] = np.log1p(test['distance'])


# Plotting the distribution of "distance" on test dataset

# In[88]:


#rechecking the distribution for distance
sns.distplot(test['distance'],bins='auto',color='red')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# Observation: the "distance" distribution is not skewed and requires no further transformation.

# In[89]:


train.head()


# # Preparing the input features and target label matrices

# In[90]:


X=np.array(train.iloc[:,1:])
y=np.array(train.iloc[:,0])


# In[91]:


#Splitting the dataset into train and test dataset

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# # Linear Regression Model

# In[92]:


#Training the data based on Linear Regression model
model_lr=LinearRegression()
model_lr.fit(X_train,y_train)


# In[93]:


#Predicting the model on train data
train_pred_lr=model_lr.predict(X_train)


# In[94]:


test_pred_lr=model_lr.predict(X_test)


# In[95]:


##calculating RMSE for test data
test_rmse_lr = np.sqrt(mean_squared_error(y_test, test_pred_lr))

##calculating RMSE for train data
train_rmse_lr= np.sqrt(mean_squared_error(y_train, train_pred_lr))


# In[96]:


print("Root Mean Squared Error For Training data = ",train_rmse_lr)
print("Root Mean Squared Error For Test data = ",test_rmse_lr)


# In[97]:


print("R2 score for training data is",r2_score(y_train,train_pred_lr))
print("R2 score for testing data is",r2_score(y_test,test_pred_lr))


# # Decision Tree Model

# In[98]:


#Training the data using Decision Tree model
model_dt=DecisionTreeRegressor(max_depth=2)
model_dt.fit(X_train,y_train)
train_pred_dt=model_dt.predict(X_train)
test_pred_dt=model_dt.predict(X_test)


# In[99]:


##calculating RMSE for test data
test_rmse_dt = np.sqrt(mean_squared_error(y_test, test_pred_dt))
##calculating RMSE for train data
train_rmse_dt= np.sqrt(mean_squared_error(y_train, train_pred_dt))


# In[100]:


print("Root Mean Squared Error For Training data = ",train_rmse_dt)
print("Root Mean Squared Error For Test data = ",test_rmse_dt)


# In[101]:


print("R2 score for training data is",r2_score(y_train,train_pred_dt))
print("R2 score for testing data is",r2_score(y_test,test_pred_dt))


# # Random Forest Regressor Model

# In[102]:


#Training the data using Random Forest Regressor
model_rf=RandomForestRegressor(n_estimators=101)  #n_esimators means No.of Trees
model_rf.fit(X_train,y_train)
train_pred_rf=model_rf.predict(X_train)
test_pred_rf=model_rf.predict(X_test)


# In[103]:


##calculating RMSE for test data
test_rmse_rf = np.sqrt(mean_squared_error(y_test, test_pred_rf))
##calculating RMSE for train data
train_rmse_rf= np.sqrt(mean_squared_error(y_train, train_pred_rf))


# In[104]:


print("Root Mean Squared Error For Training data = ",train_rmse_rf)
print("Root Mean Squared Error For Test data = ",test_rmse_rf)


# In[105]:


print("R2 score for training data is",r2_score(y_train,train_pred_rf))
print("R2 score for testing data is",r2_score(y_test,test_pred_rf))


# # Gradient Boosting Regressor Model

# In[106]:


#training the data using Gradient Boosting model
model_gb=GradientBoostingRegressor()
model_gb.fit(X_train,y_train)
train_pred_gb=model_gb.predict(X_train)
test_pred_gb=model_gb.predict(X_test)


# In[107]:


##calculating RMSE for test data
test_rmse_gb = np.sqrt(mean_squared_error(y_test, test_pred_gb))

##calculating RMSE for train data
train_rmse_gb= np.sqrt(mean_squared_error(y_train, train_pred_gb))


# In[108]:


print("Root Mean Squared Error For Training data = ",train_rmse_gb)
print("Root Mean Squared Error For Test data = ",test_rmse_gb)


# In[109]:


print("R2 score for training data is",r2_score(y_train,train_pred_gb))
print("R2 score for testing data is",r2_score(y_test,test_pred_gb))


# # Hyperparameter tuning

# Parameter tuning for Random Forest model

# In[110]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42) # If we don't give random_state(0 to 42) whenever we execute each time new value generated ;
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[111]:


from sklearn.model_selection import train_test_split,RandomizedSearchCV
##Random Search CV on Random Forest Model

model_rrf = RandomForestRegressor(random_state = 0) #rrf=Rondom forest regressor
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))

# Create the random grid
rand_grid = {'n_estimators': n_estimator,
               'max_depth': depth}

randomcv_rf = RandomizedSearchCV(model_rrf, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)#cv=changed version
randomcv_rf = randomcv_rf.fit(X_train,y_train)
predictions_RRF = randomcv_rf.predict(X_test)

view_best_params_RRF = randomcv_rf.best_params_

best_model = randomcv_rf.best_estimator_

predictions_RRF = best_model.predict(X_test)

#R^2
RRF_r2 = r2_score(y_test, predictions_RRF)
#Calculating RMSE
RRF_rmse = np.sqrt(mean_squared_error(y_test,predictions_RRF))

print('Random Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_RRF)
print('R-squared = {:0.2}.'.format(RRF_r2))
print('RMSE = ',RRF_rmse)


# Parameter tuning for Gradient Boosting Regressor

# In[112]:


gb = GradientBoostingRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(gb.get_params())


# In[113]:


##Random Search CV on gradient boosting model

model_gbr = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))

# Create the random grid
rand_grid = {'n_estimators': n_estimator,
               'max_depth': depth}

randomcv_gb = RandomizedSearchCV(gb, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_gb = randomcv_gb.fit(X_train,y_train)
predictions_gb = randomcv_gb.predict(X_test)

view_best_params_gb = randomcv_gb.best_params_

best_model = randomcv_gb.best_estimator_

predictions_gb = best_model.predict(X_test)

#R^2
gb_r2 = r2_score(y_test, predictions_gb)
#Calculating RMSE
gb_rmse = np.sqrt(mean_squared_error(y_test,predictions_gb))

print('Random Search CV Gradient Boosting Model Performance:')
print('Best Parameters = ',view_best_params_gb)
print('R-squared = {:0.2}.'.format(gb_r2))
print('RMSE = ', gb_rmse)


# Final parameter extraction for Random Forest Model

# In[114]:


from sklearn.model_selection import GridSearchCV    
## Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(X_train,y_train)
view_best_params_GRF = gridcv_rf.best_params_

#Apply model on test data
predictions_grf = gridcv_rf.predict(X_test)

#R^2
grf_r2 = r2_score(y_test, predictions_grf)
#Calculating RMSE
grf_rmse = np.sqrt(mean_squared_error(y_test,predictions_grf))

print('Grid Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_GRF)
print('R-squared = {:0.2}.'.format(grf_r2))
print('RMSE = ',(grf_rmse))


# Final parameter extraction for Gradient Boosting Model

# In[115]:


## Grid Search CV for gradinet boosting
gb = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_gb = GridSearchCV(gb, param_grid = grid_search, cv = 5)
gridcv_gb = gridcv_gb.fit(X_train,y_train)
view_best_params_Ggb = gridcv_gb.best_params_

#Apply model on test data
predictions_Ggb = gridcv_gb.predict(X_test)

#R^2
Ggb_r2 = r2_score(y_test, predictions_Ggb)
#Calculating RMSE
Ggb_rmse = np.sqrt(mean_squared_error(y_test,predictions_Ggb))

print('Grid Search CV Gradient Boosting regression Model Performance:')
print('Best Parameters = ',view_best_params_Ggb)
print('R-squared = {:0.2}.'.format(Ggb_r2))
print('RMSE = ',(Ggb_rmse))


# Observations: the final parameters were selected for both the random forest and Gradient Boosting models, the accuracy for both the models is same yet the error for Random forest model is less than that of the Gradient Boosting model.
#     Hence we choose the Random Forest Model to predict the values for the "test.csv"

# # Selection of the model: Random Forest Regressor Model

# In[116]:


## Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(X_train,y_train)
view_best_params_GRF = gridcv_rf.best_params_

#Apply model on test data
predictions_grf_test_Df = gridcv_rf.predict(test)


# # Creating the target label on the "test"(test.csv)

# In[117]:


predictions_grf_test_Df
test['Predicted_fare'] = predictions_grf_test_Df


# In[118]:


test.head()


# Writing the whole dataframe into "test.csv"

# In[119]:


test.to_csv('test.csv')


# In[ ]:




