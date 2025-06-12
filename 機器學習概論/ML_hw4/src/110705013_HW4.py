# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # This is the SVM classifier class from sklearn.

# The two global variables below are the hyperparameters of the SVM with polynomial and rbf kernels.
# You should tune them to achieve the best performance on the test set.
degree_ = 4 # degree_ should be an integer.
gamma_ = 1.0

# Compute and return the Gram matrix of the given data.
def gram_matrix(X1, X2, kernel_function):
    n_samples1 = X1.shape[0]
    n_samples2 = X2.shape[0]
    gram = np.zeros((n_samples1, n_samples2))

    for i in range(n_samples1):
        for j in range(n_samples2):
            gram[i][j] = kernel_function(X1[i], X2[j])

    return gram

# Compute and return the linear kernel between two vectors.
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Compute and return the polynomial kernel between two vectors.
def polynomial_kernel(x1, x2, degree=degree_):
    return (np.dot(x1, x2) + 1) ** degree

# Compute and return the rbf kernel between two vectors.  
def rbf_kernel(x1, x2, gamma=gamma_):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# Do not modify the main function architecture.
# You can only modify the hyperparameter C of the SVC class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    # Tune the hyperparameter (C) of the SVM with your linear kernel here.
    C_ = 3.0
    svc = SVC(kernel='precomputed', C=C_)
    svc.fit(gram_matrix(X_train, X_train, linear_kernel), y_train)
    y_pred = svc.predict(gram_matrix(X_test, X_train, linear_kernel))
    print(f"Accuracy of using linear kernel (C = {C_}): ", accuracy_score(y_test, y_pred))
    assert accuracy_score(y_test, y_pred) > 0.8

    # Tune the hyperparameter (degree) by the global variable degree_ of your polynomial_kernel function.
    # Tune the hyperparameter (C) of the SVM with your polynomial kernel here.
    C_ = 0.1
    svc = SVC(kernel='precomputed', C=C_)
    svc.fit(gram_matrix(X_train, X_train, polynomial_kernel), y_train)
    y_pred = svc.predict(gram_matrix(X_test, X_train, polynomial_kernel))
    print(f"Accuracy of using polynomial kernel (C = {C_}, degree = {degree_}): ", accuracy_score(y_test, y_pred))

    # Tune the hyperparameter (gamma) by the global variable gamma_ of your rbf_kernel function.
    # Tune the hyperparameter (C) of the SVM with your rbf kernel here.
    C_ = 3.0
    svc = SVC(kernel='precomputed', C=C_)
    svc.fit(gram_matrix(X_train, X_train, rbf_kernel), y_train)
    y_pred = svc.predict(gram_matrix(X_test, X_train, rbf_kernel))
    print(f"Accuracy of using rbf kernel (C = {C_}, gamma = {gamma_}): ", accuracy_score(y_test, y_pred))