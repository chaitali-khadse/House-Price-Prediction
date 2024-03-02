#%% Cell 1 : Importing necessary libraries for data manipulation, analysis, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Cell 2 Loading the housing dataset and dropping any rows with missing values
data = pd.read_csv("housing.csv")
data.dropna(inplace=True)


#%% Cell 3 Splitting the dataset into features (X) and the target variable (y)
# 'median_house_value' is the target variable, and the rest are features
from sklearn.model_selection import train_test_split

X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']
# Splitting the dataset into training and testing sets with a test size of 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%% Cell 4 Joining the training features and target variable into a single DataFrame for analysis
train_data = X_train.join(y_train)

#%% Cell 5 Generating histograms for all columns in the training data
# This initial attempt includes non-numeric columns which can cause errors
train_data.hist()
# Modified to exclude non-numeric columns and generate histograms only for numeric data
# Cell 6 
train_data.select_dtypes(include=[np.number]).hist()

# %% Cell 7 Generating histograms with a specified figure size for better visibility
train_data.hist(figsize=(15,8))

# %% # Encoding categorical variables and generating a correlation heatmap
# This helps in understanding the relationship between different features
train_data_encoded = pd.get_dummies(train_data)
sns.heatmap(train_data_encoded.corr(), annot=True, cmap="YlGnBu")

# %% Applying logarithmic transformations to certain features to potentially normalize their distributions
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

# %% Generating histograms again to visualize the distribution after logarithmic transformation

train_data.hist(figsize=(15,8))
# %% Encoding the 'ocean_proximity' categorical variable using one-hot encoding and dropping the original column

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
# %% Calculating correlations again with the encoded data
numeric_data = train_data.select_dtypes(include=[np.number])
plt.figure(figsize=(15,8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="YlGnBu")

# %% Visualizing the geographical distribution of median house values using a scatter plot

plt.figure(figsize=(15,8))
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")
# %%  Creating new features that might help in improving model performance
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']
# Calculating and visualizing the correlation matrix with the new features included

plt.figure(figsize=(15,8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="YlGnBu")

# %%# Preparing data for linear regression model training
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']

reg = LinearRegression()
#%% Fitting the linear regression model
reg.fit(X_train, y_train)
# %%
test_data = X_test.join(y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)


test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

# %%
X_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']
# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Assuming 'ocean_proximity' is the only categorical column needing encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']),
        ('cat', OneHotEncoder(), ['ocean_proximity'])
    ])

# Define the pipeline
reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', LinearRegression())])

# Split the data
X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
reg_pipeline.fit(X_train, y_train)

# %%
print(reg_pipeline.score(X_test, y_test))
# %%
