import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def train_and_save_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Save the model
    with open("iris_knn_model.pkl", "wb") as f:
        pickle.dump(knn, f)
    
    print("Model trained and saved as 'iris_knn_model.pkl'.")

if __name__ == "__main__":
    train_and_save_model() 
