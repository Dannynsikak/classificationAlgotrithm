import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

file_path = "Titanic.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns (e.g., 'who' is redundant with 'sex', 'class' is redundant with ticket class)
df = df.drop(columns=['who'])

# Encode categorical variables
label_encoders = {}
for col in ['sex', 'embarked', 'class', 'alone']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Handle missing values by filling with the median (for age)
df.loc[:, 'age'] = df['age'].fillna(df['age'].median())

# Define features and target variable
X = df.drop(columns=['survived'])
y = df['survived']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy, report

# Print evaluation results
print(f"Model Accuracy: {accuracy:.2f}")  # Prints accuracy with 2 decimal places
print("\nClassification Report:\n")
print(report)