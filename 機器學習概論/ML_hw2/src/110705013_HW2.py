# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        np.random.seed(24)
        self.weights = np.random.rand(len(X[0]))
        self.intercept = 0
        for i in range(self.iteration):
            value = np.matmul(X, self.weights)
            value = value + self.intercept
            prediction = self.sigmoid(value)
            grad = prediction - y
            gradw = grad[:,np.newaxis] * X 
            gradw = np.mean(gradw, axis = 0)
            gradi = grad.mean()
            self.weights = self.weights - self.learning_rate * gradw
            self.intercept = self.intercept - self.learning_rate * gradi
            
            if(i==125000):
                self.learning_rate /= 2




            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        value = np.matmul(X, self.weights)
        value = value + self.intercept
        prediction = self.sigmoid(value)
        prediction[prediction > 0.5] = int(1)
        prediction[prediction <= 0.5 ] = int(0)
        return prediction


    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / ( 1 + np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        x1 = X[y==1]
        x0 = X[y==0]

        self.m0 = np.mean(x0, axis=0)
        self.m1 = np.mean(x1, axis=0)
        self.sw = np.cov(x0, rowvar=False) + np.cov(x1, rowvar=False)
        
        diff = (self.m1-self.m0).reshape(2,1)
        self.sb = np.matmul(diff , diff.transpose())
        # print("sb: ", self.sb)

        self.w = np.matmul(np.linalg.inv(self.sw), diff).reshape(2)
        self.slope = self.w[1] / self.w[0]
        #self.plot_projection(X,y)

        # Choose the eigenvector corresponding to the largest eigenvalue
        # self.w = eigenvectors[:, -1]

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        value0 = np.dot(self.w, self.m0)
        value1 = np.dot(self.w, self.m1)
        
        predicted = np.matmul(X, self.w)
        diff_0 = np.abs(predicted - value0)
        diff_1 = np.abs(predicted - value1)

        to0 = diff_0 < diff_1

        # Set elements closer to 5 to 0, and elements closer to 50 to 1
        result_array = np.where(to0, 0, 1)
        return result_array

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X,y):
        # set the figure size
        plt.figure(figsize=(7,7))  

        # devide values
        x_values = X[:, 0].reshape((X.shape)[0],1)
        y_values = X[:, 1].reshape((X.shape)[0],1)

        # predict the result
        Y = self.predict(X)

        # split into 2 type
        x_values_0 = x_values[Y == 0]
        y_values_0 = y_values[Y == 0]
        x_values_1 = x_values[Y == 1]
        y_values_1 = y_values[Y == 1]
        
        # draw the line
        intercept = 387
        line_x = np.linspace(40, 140, 100)
        line_y = line_x * self.slope + intercept
        plt.plot(line_x, line_y, color='k', linestyle='-', linewidth=1)

        # draw the projection point
        projected_points = np.empty_like(X)
        for i in range(len(X)):
            x, y = X[i]
            x_proj = ( x + self.slope *y  - self.slope * intercept) / (self.slope**2 +1 )
            y_proj = self.slope * x_proj + intercept
            projected_points[i] = [x_proj, y_proj]
            if(Y[i]):
                plt.plot([x_proj,X[i][0]], [y_proj,X[i][1]],linestyle='-',color='b', linewidth=1,alpha=0.2)
                plt.scatter(x_proj, y_proj, c='b', marker='o', s=2)

            else:
                plt.plot([x_proj,X[i][0]], [y_proj,X[i][1]],linestyle='-',color='r', linewidth=1,alpha=0.2)
                plt.scatter(x_proj, y_proj, c='r', marker='o', s=2)

        # limit plot range
        plt.xlim(-20, 140)
        plt.ylim(60,220)

        # draw data points in X
        plt.scatter(x_values_0, y_values_0, c='r', marker='o', s=2)
        plt.scatter(x_values_1, y_values_1, c='b', marker='o', s=2)
        plt.xlabel('age')
        plt.ylabel('thalach')
        plt.legend()
        plt.title(f"Projection Line - slope: {self.slope}, intercept: {intercept}")
        plt.show()
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.001, iteration=130000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"

