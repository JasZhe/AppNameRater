import data_parser
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

app_title, app_rating = data_parser.get_app_info()

titles_train, titles_test, y_train, y_test = train_test_split(app_title, app_rating, test_size=0.25, random_state=42)

vectorizer = CountVectorizer(min_df=20, lowercase=True)
vectorizer.fit(app_title)

X_train = vectorizer.transform(titles_train)
X_test = vectorizer.transform(titles_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# About the same as the neural network
print(classifier.score(X_test, y_test))
