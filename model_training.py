# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Prepare features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'diabetes_model.pkl'")
