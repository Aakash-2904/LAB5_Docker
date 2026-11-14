# Import necessary libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

if __name__ == '__main__':
    # Load the Wine dataset (178 samples, 13 features, 3 cultivar classes)
    wine = load_wine()
    X, y = wine.data, wine.target

    # Split the data into training and testing sets (stratified to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))

    # Save the model to a file
    joblib.dump(model, 'wine_model.pkl')

    print("The model training was successful")
