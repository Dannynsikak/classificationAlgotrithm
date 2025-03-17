import pandas as pd

# Load the dataset
file_path = "spam_ham_dataset.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
df.info(), df.head()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Drop unnecessary column
df = df.drop(columns=["Unnamed: 0"])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label_num"], test_size=0.2, random_state=42)

# Convert text data to numerical using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy, report

# Print evaluation results
print(f"Model Accuracy: {accuracy:.2f}")  # Prints accuracy with 2 decimal places
print("\nClassification Report:\n")
print(report)