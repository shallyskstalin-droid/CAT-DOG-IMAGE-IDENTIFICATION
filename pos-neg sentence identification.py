from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
texts = [
    "I love this product",
    "This is amazing",
    "Very happy with the service",
    "Worst experience ever",
    "I hate this product",
    "Terrible support"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = Positive, 0 = Negative
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
sample_text = ["I am very disappointed"]
prediction = model.predict(vectorizer.transform(sample_text))

if prediction[0] == 1:
    print("Sentiment: POSITIVE 😊")
else:
    print("Sentiment: NEGATIVE 😞")
