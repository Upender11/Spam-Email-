import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\JOb\mail_data.csv")
print(data)

# Clean null values
mail = data.where(pd.notnull(data), "")
print(mail.head())
print(mail.shape)

# Encode labels: Spam = 0, Ham = 1
mail['Category'] = mail['Category'].map({'spam': 0, 'ham': 1})
# If values are 'Spam'/'Ham' (with capital), use this instead:
# mail['Category'] = mail['Category'].str.lower().map({'spam': 0, 'ham': 1})

# Separate features and labels
X = mail['Message']
Y = mail['Category']

print(X)
print(Y)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

# Feature extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
x_test_features = feature_extraction.transform(X_test)

print(X_train)
print(Y_train)
print(X_train_features)

# Train model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Accuracy on training data
prediction_on_data = model.predict(X_train_features)
accuracy_on_data = accuracy_score(Y_train, prediction_on_data)
print('Accuracy on training data:', accuracy_on_data)

# Accuracy on test data
prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data:', accuracy_on_test_data)

# Predictive system
input_mail = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]
input_data = feature_extraction.transform(input_mail)
prediction = model.predict(input_data)

print(prediction)
if prediction[0] == 1:
    print("Ham Mail")
else:
    print("Spam Mail")
