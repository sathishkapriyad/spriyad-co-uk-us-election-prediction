import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load datasets
sp_president_df = pd.read_csv('1976-2020-president.csv', encoding='latin1')
sp_senate_df = pd.read_csv('1976-2020-senate.csv', encoding='latin1')

# Handle missing values
sp_president_df.fillna(method='ffill', inplace=True)
sp_senate_df.fillna(method='ffill', inplace=True)

# Combine datasets to fit the LabelEncoder with all possible values
combined_df = pd.concat([sp_president_df, sp_senate_df])

# Encode categorical variables
le_state = LabelEncoder()
le_candidate = LabelEncoder()
le_party_detailed = LabelEncoder()

combined_df['state'] = le_state.fit_transform(combined_df['state'])
combined_df['candidate'] = le_candidate.fit_transform(combined_df['candidate'])
combined_df['party_detailed'] = le_party_detailed.fit_transform(combined_df['party_detailed'])

# Save the LabelEncoders
joblib.dump(le_state, 'le_state.pkl')
joblib.dump(le_candidate, 'le_candidate.pkl')
joblib.dump(le_party_detailed, 'le_party_detailed.pkl')

# Apply the transformations to the original datasets
sp_president_df['state'] = le_state.transform(sp_president_df['state'])
sp_president_df['candidate'] = le_candidate.transform(sp_president_df['candidate'])
sp_president_df['party_detailed'] = le_party_detailed.transform(sp_president_df['party_detailed'])

sp_senate_df['state'] = le_state.transform(sp_senate_df['state'])
sp_senate_df['candidate'] = le_candidate.transform(sp_senate_df['candidate'])
sp_senate_df['party_detailed'] = le_party_detailed.transform(sp_senate_df['party_detailed'])

# Select relevant features
sp_features = ['year', 'state', 'candidate', 'party_detailed', 'totalvotes']
sp_target = 'party_simplified'

# Create a combined dataset
sp_election_dataset = pd.concat([sp_president_df[sp_features + [sp_target]], sp_senate_df[sp_features + [sp_target]]])

# Split the data
X = sp_election_dataset[sp_features]
y = sp_election_dataset[sp_target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
sp_model = RandomForestClassifier(n_estimators=100, random_state=42)
sp_model.fit(X_train, y_train)

# Evaluate the model
y_pred = sp_model.predict(X_test)
sp_accuracy = accuracy_score(y_test, y_pred)
sp_classification_report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {sp_accuracy}")
print("Classification Report:")
print(sp_classification_report)

# Save the model
joblib.dump(sp_model, 'sp_election_model.pkl')
