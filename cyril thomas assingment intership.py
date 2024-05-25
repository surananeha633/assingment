#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages in python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[ ]:


#Write a web scraping script which takes an input of any film actor and gives the output of filmography of that actor in descending order.
#Use web scraping method
#Eg : 
#input [Leonardo DiCaprio]
#Output : Films done by Leonardo DiCaprio in descending order


# In[2]:


pip install requests beautifulsoup4


# In[3]:


import requests
from bs4 import BeautifulSoup
import re

def get_actor_filmography(actor_name):
    # Convert the actor name to a format suitable for URL
    actor_name = actor_name.replace(' ', '_')
    
    # Wikipedia URL for the actor
    url = f"https://en.wikipedia.org/wiki/{actor_name}"
    
    # Send a request to fetch the page content
    response = requests.get(url)
    
    # Check if the page was fetched successfully
    if response.status_code != 200:
        print("Error fetching the Wikipedia page.")
        return
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the filmography section
    filmography_header = soup.find(id=re.compile("Filmography|filmography", re.IGNORECASE))
    if not filmography_header:
        print("Filmography section not found.")
        return
    
    # Get the parent element of the filmography section
    filmography_section = filmography_header.find_parent(['h2', 'h3', 'h4', 'h5', 'h6'])

    # Traverse the sibling elements to collect all filmography entries
    films = []
    for sibling in filmography_section.find_next_siblings():
        # Stop if a new section starts
        if sibling.name in ['h2', 'h3', 'h4', 'h5', 'h6']:
            break
        # Find film entries (typically they are in tables or lists)
        if sibling.name == 'ul':
            for li in sibling.find_all('li'):
                films.append(li.get_text())
        elif sibling.name == 'table':
            for row in sibling.find_all('tr')[1:]:  # Skip the header row
                cols = row.find_all('td')
                if len(cols) > 1:
                    films.append(cols[0].get_text().strip())

    # Sort the films in descending order based on year (assuming year is at the start of each entry)
    def extract_year(film):
        match = re.search(r'\b(19|20)\d{2}\b', film)
        return int(match.group(0)) if match else 0

    films.sort(key=extract_year, reverse=True)
    
    # Print the films
    for film in films:
        print(film)

# Input
actor_name = input("Enter the name of the actor: ")
get_actor_filmography(actor_name)


# In[ ]:


#Given a list of planets discovered by KEPLER.
#Kepler Data: https://drive.google.com/drive/folders/1GwqC4STc_KgVPofacQUzKHBMHQsmflvY?usp=sharing
#Create an ML algorithm to classify the planets as Candidate/False positive/Confirmed etc based on the  column “koi_disposition”.


# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = 'path_to_your_kepler_data.csv'
data = pd.read_csv(file_path)

# Inspect the dataset
print(data.head())
print(data.info())
print(data['koi_disposition'].value_counts())

# Data Preprocessing
# Drop columns with too many missing values or not useful for prediction
data = data.drop(['koi_teq_err1', 'koi_teq_err2'], axis=1)

# Fill missing values with mean/mode/median
data = data.fillna(data.mean())

# Encode the target variable
data['koi_disposition'] = data['koi_disposition'].map({
    'FALSE POSITIVE': 0,
    'CANDIDATE': 1,
    'CONFIRMED': 2
})

# Select features and target variable
X = data.drop(['koi_disposition'], axis=1)
y = data['koi_disposition']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model for future use
import joblib
joblib.dump(model, 'kepler_planet_classifier.pkl')


# In[ ]:


#Why did you choose the particular algorithm?
#answer:
*Support Vector Machines (SVM)*: While SVMs can be very powerful, they often require careful tuning of hyperparameters and can be computationally intensive, especially with large datasets. They also struggle with missing values and require extensive preprocessing.
*Neural Networks*: Neural networks are powerful but require a large amount of data and computational power to train effectively. They also require significant preprocessing and tuning to perform well.
*Gradient Boosting Machines (GBM)*: GBM can provide higher accuracy but at the cost of longer training times and more complexity in hyperparameter tuning. They also can be more prone to overfitting if not carefully managed.
*K-Nearest Neighbors (KNN)*: KNN is simple and easy to understand but can be very slow with large datasets and high-dimensional data. It also struggles with missing values and requires scaling of features


# In[ ]:


#What are the different tuning methods used for the algorithm?
#answer:
### 1. *Support Vector Machines (SVM)*
- *Pros*: SVMs are effective in high-dimensional spaces and are versatile in terms of kernel functions.
- *Cons*: They can be computationally expensive, especially with large datasets, and require careful tuning of hyperparameters. They also struggle with missing values.
- *Rationale*: SVMs can be very accurate for binary classification problems but might not scale well with the size and complexity of the Kepler dataset.

### 2. *Gradient Boosting Machines (GBM)*
- *Pros*: Gradient Boosting is known for its high predictive performance and ability to handle a variety of data types and structures.
- *Cons*: They are more prone to overfitting if not properly tuned, and training can be slower compared to Random Forests.
- *Rationale*: GBMs, including implementations like XGBoost, LightGBM, and CatBoost, often provide state-of-the-art performance but require careful tuning and longer training times.

### 3. *Neural Networks*
- *Pros*: Neural networks, especially deep learning models, can capture complex patterns in data and are very flexible.
- *Cons*: They require large amounts of data, significant computational resources, and extensive hyperparameter tuning. They are also less interpretable compared to tree-based models.
- *Rationale*: Neural networks could potentially capture complex relationships in the Kepler data but might be overkill and harder to interpret for this task.

### 4. *K-Nearest Neighbors (KNN)*
- *Pros*: KNN is simple and easy to implement, with no training phase required.
- *Cons*: It can be very slow for large datasets and high-dimensional data, and it is sensitive to the choice of the number of neighbors (k) and distance metric.
- *Rationale*: KNN might not perform well with the size and complexity of the Kepler dataset due to its computational inefficiency and sensitivity to scaling.

### 5. *Logistic Regression*
- *Pros*: Logistic Regression is simple, easy to implement, and provides probabilities for classification.
- *Cons*: It assumes linear relationships between features and the target, which might not be sufficient for complex datasets.
- *Rationale*: Logistic Regression could serve as a good baseline model, but it might not capture complex patterns in the Kepler data.

### 6. *Decision Trees*
- *Pros*: Simple to understand and interpret, and can handle both numerical and categorical data.
- *Cons*: Prone to overfitting and can have high variance.
- *Rationale


# In[ ]:


#What is the accuracy?
#answer:
    ### Step-by-Step Example:

1. *Data Preparation and Model Training*:
python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = 'path_to_your_kepler_data.csv'
data = pd.read_csv(file_path)

# Inspect the dataset
print(data.head())
print(data.info())
print(data['koi_disposition'].value_counts())

# Data Preprocessing
data = data.fillna(data.mean())  # Handle missing values

# Encode the target variable
data['koi_disposition'] = data['koi_disposition'].map({
    'FALSE POSITIVE': 0,
    'CANDIDATE': 1,
    'CONFIRMED': 2
})

# Select features and target variable
X = data.drop(['koi_disposition'], axis=1)
y = data['koi_disposition']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


2. *Model Evaluation*:
python
# Model Evaluation
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Detailed classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


### Example Output:
plaintext
Accuracy: 0.90  # This is a hypothetical value for illustration purposes


### Detailed Explanation:

- *Accuracy*: The ratio of correctly predicted instances to the total instances in the test set. It provides a quick overview of how well the model is performing but doesn't give insights into individual class performance, especially in cases of imbalanced datasets.
- *Classification Report*: Provides detailed metrics for each class, including precision, recall, and F1-score.
- *Confusion Matrix*: Shows the number of true positive, false positive, true negative, and false negative predictions, giving more insight into the model's performance for each class.

### Note:
The actual accuracy will depend on several factors, including the quality of the data, the presence of missing values, the features used, and how well the hyperparameters are tuned. In practice, you might find the accuracy of the Random Forest model to be around 85-90%, but this can vary. It’s essential to also consider other metrics like precision, recall, and F1-score, especially if the classes are imbalanced.

### Further Steps:
- *Hyperparameter Tuning*: Use Grid Search, Random Search, or Bayesian Optimization to find the best hyperparameters for improved accuracy.
- *Cross-Validation*: Use cross-validation to ensure the model's performance is consistent and not just a result of a particular train-test split.

By following these steps, you can determine the accuracy of your Random Forest model on the Kepler dataset

