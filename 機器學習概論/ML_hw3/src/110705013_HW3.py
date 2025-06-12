# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    if(len(y)== 0):
        return 1
    unique_labels, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    gini = 1 - np.sum(proportions ** 2)
    
    return gini

# This function computes the entropy of a label array.
def entropy(y):
    unique_labels, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    entropy = -np.sum(proportions * np.log2(proportions))
    return entropy


class Node:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold  # Threshold value for the feature
        self.value = value  # Predicted value if node is a leaf node
        self.left = left  # Left child node
        self.right = right  # Right child node

# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.tree = None
        self.columns = ["age", "sex", "cp", "fbs", "thalach", "thal"]
        self.count = [0,0,0,0,0,0]
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    def split(self, X, y, feature_index, threshold):
        '''
        Input
            X : data
            y : label
            feature_index
            threshold
        Output
            corresponding X & Y for left & right node
        '''
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        left_y = y[left_indices]
        right_y = y[right_indices]
        left_X = X[left_indices]
        right_X = X[right_indices]
        
        return left_X, right_X, left_y, right_y

    def find_best_split(self, X, y):
        '''
        Input 
            X : data
            y : label
        Output the best feature & threshold
        '''
        best_impurity = float('inf')
        best_feature_index = None
        best_threshold = None

        # for features and all the values
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # split according to values and find the minimum impurity
                _, _, left_y, right_y = self.split(X, y, feature_index, threshold)
                impurity_left = self.impurity(left_y)
                impurity_right = self.impurity(right_y)
                
                total_impurity = (len(left_y) * impurity_left + len(right_y) * impurity_right) / len(y)
                #print(total_impurity)
                if total_impurity < best_impurity:
                    best_impurity = total_impurity
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold

    def build_tree(self, X, y, depth):
        '''
        build the tree recursively
        '''
        # assign value if leaf node or only one class
        if len(y) == 0:
            return Node(value=1)
        if depth == 0 or len(np.unique(y)) == 1:
            # add 2 since i have label -1 in adaboost and i use bicount
            y = y+2
            value = np.argmax(np.bincount(y))
            # minux it back
            value = value-2
            return Node(value=value)

        # find best feature & threshold
        best_feature_index, best_threshold = self.find_best_split(X, y)
        self.count[best_feature_index] += 1
        
        if best_feature_index is None or best_threshold is None:
            return Node(value=np.argmax(np.bincount(y)))

        # split X and Y
        left_X, right_X, left_Y, right_Y = self.split(X, y, best_feature_index, best_threshold)

        # build tree recursively
        left_node = self.build_tree(left_X, left_Y, depth - 1)
        right_node = self.build_tree(right_X, right_Y, depth - 1)

        # return current node
        return Node(feature_index=best_feature_index, threshold=best_threshold, left=left_node, right=right_node)


    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        
        self.tree = self.build_tree(X, y, self.max_depth)
        # self.plot_feature_importance_img(self.columns)
    
    def predict_sample(self, node, x):
        # leaf node -> return value
        if node.value is not None:
            return node.value
        
        # recursively trace down the tree
        if x[node.feature_index] <= node.threshold:
            return self.predict_sample(node.left, x)
        else:
            return self.predict_sample(node.right, x)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self.predict_sample(self.tree, sample)
            predictions.append(prediction)
        return np.array(predictions)
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        plt.figure(figsize=(8, 6))
        plt.barh(columns, self.count, color='#1f77b4')
        plt.title('Feature importance')
        plt.tight_layout()
        plt.show()


# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=20):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.alphas = []  # List to store alphas for weak classifiers
        self.classifiers = []  # List to store weak classifiers

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        # know number of entry in X and initialize D
        n_samples = X.shape[0]
        D = np.full(n_samples, (1 / n_samples))

        zero_indices = y[:] == 0
        y[zero_indices] = -1

        for _ in range(self.n_estimators):
            rows = np.random.choice(len(X), len(X), p=D)
            dataset = np.array([X[row] for row in rows])
            labels  = np.array([y[row] for row in rows])

            tree = DecisionTree(criterion=self.criterion, max_depth=1)
            tree.fit(dataset, labels)
            predictions = tree.predict(X)
            
            err = np.sum(D * (predictions != y)) / np.sum(D) 
            alpha = 0.5 * np.log((1 - err) / err)
            D *= np.exp(-alpha * y * predictions)
            D /= np.sum(D)

            self.alphas.append(alpha)
            self.classifiers.append(tree)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        n_samples = X.shape[0]
        classes = np.zeros((n_samples, len(self.classifiers)))

        for i, clf in enumerate(self.classifiers):
            classes[:, i] = self.alphas[i] * clf.predict(X)  # Weighted predictions

        classes = np.sum(classes, axis=1)
        minus_one = classes[:] <= 0
        one = classes[:] > 0
        classes[minus_one] = 0
        classes[one] = 1
        return classes


# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
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

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(2000)

    # Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))

    # AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.

    ada = AdaBoost(criterion='gini', n_estimators=20)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))


    
