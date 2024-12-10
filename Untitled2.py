#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[24]:


data = pd.read_csv('gym_members_exercise_tracking_synthetic_data.csv')


# In[25]:


data.head()


# In[7]:


print(data.info())


# In[8]:


print(data.describe())


# In[9]:


print(data.isnull().sum())


# In[10]:


num_cols = data.select_dtypes(include=['float64']).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())
cat_cols = data.select_dtypes(include=['object']).columns
data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
data['Max_BPM'] = pd.to_numeric(data['Max_BPM'], errors='coerce')
data['Max_BPM'] = data['Max_BPM'].fillna(data['Max_BPM'].median())


# In[11]:


missing_values = data.isnull().sum()
print("Missing Values after Cleaning:")
print(missing_values)


# In[12]:


sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(14, 10))

num_features = [
    "Age", "Weight (kg)", "Height (m)", "BMI", "Calories_Burned",
    "Session_Duration (hours)", "Fat_Percentage", "Water_Intake (liters)"
]
for i, feature in enumerate(num_features, 1):
    plt.subplot(4, 2, i)
    sns.histplot(data[feature], kde=True, bins=30, color="skyblue")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# In[13]:


plt.figure(figsize=(14, 10))

for i, feature in enumerate(num_features, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(data=data[feature], color="orange")
    plt.title(f"Boxplot of {feature}")
    plt.xlabel(feature)

plt.tight_layout()
plt.show()


# In[14]:


numerical_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# In[15]:


categorical_features = ["Gender", "Workout_Type"]
plt.figure(figsize=(12, 5))

for i, feature in enumerate(categorical_features, 1):
    plt.subplot(1, 2, i)
    sns.countplot(data=data, x=feature, palette="Set2")
    plt.title(f"Countplot of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# In[16]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# In[18]:


target = "Calories_Burned"
features = data.drop(columns=[target, "Gender", "Workout_Type"]) 
X = features  
y = data[target]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


dt_model = DecisionTreeRegressor(random_state=38)

# Train the model
dt_model.fit(X_train, y_train)


# In[20]:


y_pred = dt_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# In[21]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.title("Actual vs Predicted Calories Burned")
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.show()


# In[22]:


feature_importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# In[ ]:




