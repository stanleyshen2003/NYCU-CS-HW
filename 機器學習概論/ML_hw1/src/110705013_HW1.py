# You are not allowed to import any additional packages/libraries.
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        self.data = []
        
    # This function computes the closed-form solution of linear regression.
    def closed_form_fit(self, X, y):
        # Compute closed-form solution.
        # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
        ones = np.ones((X.shape[0],1))
        newX = np.hstack((ones, X))
        full = np.linalg.inv(np.matmul(np.transpose(newX) , newX))
        full = np.matmul(np.matmul(full, np.transpose(newX)), y)
        self.closed_form_weights = full[1:]
        self.closed_form_intercept = full[0]
        return

    
    # This function computes the gradient descent solution of linear regression.
    def gradient_descent_fit(self, X, y, lr, epochs):
        # Compute the solution by gradient descent.
        # Save the weights and intercept to self.gradient_descent_weights and self.gradient_descent_intercept
        self.gradient_descent_weights = np.array([1,1,1,1]).reshape((4,1))
        self.gradient_descent_intercept = np.array([0])
        N = X.shape[0]
    
        for i in range(epochs):
            for j in range(N):
                now = X[j]
                predicted = np.matmul(now,self.gradient_descent_weights) + self.gradient_descent_intercept
                gradient_withoutX = -2 / N * (y[j] - predicted)
                self.gradient_descent_weights = self.gradient_descent_weights - lr * gradient_withoutX * now.reshape((4,1))
                self.gradient_descent_intercept = self.gradient_descent_intercept - lr * gradient_withoutX
            if(i % 1000 == 0 and i>0):
                lr = lr/10
            self.data.append(self.gradient_descent_evaluate(X,y))

        self.gradient_descent_weights = self.gradient_descent_weights.reshape((4,))
        return
        

    # This function compute the MSE loss value between your prediction and ground truth.
    def get_mse_loss(self, prediction, ground_truth):
        temp = prediction - ground_truth
        temp = np.square(temp)
        loss = np.mean(temp)
        return loss

    # This function takes the input data X and predicts the y values according to your closed-form solution.
    def closed_form_predict(self, X):
        prediction = np.matmul(X, self.closed_form_weights)
        prediction = prediction + self.closed_form_intercept
        return prediction

    # This function takes the input data X and predicts the y values according to your gradient descent solution.
    def gradient_descent_predict(self, X):
        prediction = np.matmul(X, self.gradient_descent_weights.reshape((4,)))
        prediction = prediction + self.gradient_descent_intercept
        return prediction
    
    # This function takes the input data X and predicts the y values according to your closed-form solution, 
    # and return the MSE loss between the prediction and the input y values.
    def closed_form_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.closed_form_predict(X), y)

    # This function takes the input data X and predicts the y values according to your gradient descent solution, 
    # and return the MSE loss between the prediction and the input y values.
    def gradient_descent_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.gradient_descent_predict(X), y)
        
    # This function use matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_learning_curve(self):
        x = np.arange(len(self.data))
        plt.plot(x, self.data, label='Line Plot', color='b')
        plt.xlabel('X-axis (Epoch)')
        plt.ylabel('Y-axis (Loss)')
        plt.title('Loss')
        plt.legend()
        plt.show()

# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    
    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=0.1, epochs=4000)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()
    
    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    #print(f"closed form loss: {closed_form_loss}")
    #print(f"gradient descent loss: {gradient_descent_loss}")
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")
    #LR.plot_learning_curve()