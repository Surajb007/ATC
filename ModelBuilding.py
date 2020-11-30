from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class ModelBuilding():
    df = None
    vectorizer = None

    def __init__(self, dataframe):
        self.df = dataframe
        self.vectorizer = TfidfVectorizer()

    def tfidf(self, df_csv):
        X = df_csv['cleaned_data']
        y = df_csv['class']
        corpus_tfid = self.vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            corpus_tfid.toarray(), y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test, self.vectorizer

    def naiveBayes(self, X_train, y_train, X_test, y_test):
        model = naive_bayes.MultinomialNB()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = accuracy_score(list(y_test), pred)
        print('Naive Bayes')
        print(confusion_matrix(y_test, pred))
        return model, score

    def randomForest(self, X_train, y_train, X_test, y_test):
        model = RandomForestClassifier(n_estimators=200)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        print('Random Forest')
        print(confusion_matrix(y_test, pred))
        return model, score

    def xgBoost(self, X_train, y_train, X_test, y_test):
        model = XGBClassifier()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        print('XGBOOST')
        print(confusion_matrix(y_test, pred))
        return model, score
