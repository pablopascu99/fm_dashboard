from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def train_models(X_train, y_train):
    """Trains multiple models and returns them in a dictionary."""
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NaiveBayes": GaussianNB()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test, encoder):
    """Evaluates the trained models and displays a classification report."""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"Evaluation of the {name} model:")
        print(classification_report(y_test, y_pred, target_names=encoder.classes_))
        print("-" * 50)