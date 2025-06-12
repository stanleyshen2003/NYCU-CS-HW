import numpy as np
#from torch.utils.tensorboard import SummaryWriter
class Linear():
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)
        self.input = None
    def forward(self, x):
        self.input = x
        return np.dot(x, self.W)
    
    def backward(self, gradient, lr):
        gradient_out = np.dot(gradient, self.W.T)
        self.W -= lr * np.dot(self.input.T, gradient)
        return gradient_out

class Sigmoid():
    def __init__(self):
        self.output = None
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    def backward(self, gradient):
        return gradient * np.multiply(self.output, (1-self.output))
    
class ReLU():
    def __init__(self):
        self.output = None
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output
    def backward(self, gradient):
        return gradient * np.where(self.output > 0, 1, 0)
    
    
class Adagrad():
    def __init__(self, lr=0.01):
        self.lr = lr
        self.G = None
    def getlr(self, gradient):
        if self.G is None:
            self.G = 1
        self.G += np.mean(gradient**2)
        return self.lr / np.sqrt(self.G + 1e-7)
    
class SGD():
    def __init__(self, lr=0.01):
        self.lr = lr
    def getlr(self, gradient):
        return self.lr

class nn():
    def __init__(self, input_size, hidden_size, output_size, lr=0.001, epochs=100000, activation='sigmoid', optimizer='SGD', record=False):
        np.random.seed(0)
        self.L1 = Linear(input_size, hidden_size)
        self.L2 = Linear(hidden_size, hidden_size)
        self.L3 = Linear(hidden_size, output_size)
        if activation == 'sigmoid':
            self.activation1 = Sigmoid()
            self.activation2 = Sigmoid()
            self.activation3 = Sigmoid()
        elif activation == 'relu':
            self.activation1 = ReLU()
            self.activation2 = ReLU()
            self.activation3 = Sigmoid()
        if optimizer == 'SGD':
            self.optimizer1 = SGD(lr)
            self.optimizer2 = SGD(lr)
            self.optimizer3 = SGD(lr)
        if optimizer == 'Adagrad':
            self.optimizer1 = Adagrad(lr)
            self.optimizer2 = Adagrad(lr)
            self.optimizer3 = Adagrad(lr)

        self.predictions = None
        self.epochs = epochs
        self.record = record


    def forward(self, x):
        output = self.L1.forward(x)
        output = self.activation1.forward(output)
        output = self.L2.forward(output)
        output = self.activation2.forward(output)
        output = self.L3.forward(output)
        output = self.activation3.forward(output)
        self.predictions = output
        return output

    def backward(self, y):
        gradient = 2 * (self.predictions - y)
        gradient = self.activation3.backward(gradient)
        lr = self.optimizer3.getlr(gradient)
        gradient = self.L3.backward(gradient, lr)
        gradient = self.activation2.backward(gradient)
        lr = self.optimizer2.getlr(gradient)
        gradient = self.L2.backward(gradient, lr)
        gradient = self.activation1.backward(gradient)
        lr = self.optimizer1.getlr(gradient)
        gradient = self.L1.backward(gradient, lr)


    def compute_loss(self, output, y):
        return np.sum((output - y)**2)

    def train(self, x, y, output_dir):
        if self.record:
            # set directory and name for tensorboard
            writer = SummaryWriter(output_dir)

        for i in range(self.epochs):
            output = self.forward(x)
            
            '''
            enabled when no optimizer
            for j in range(x.shape[0]):
                if output[j][0] < 0.5:
                    output[j][0] = 0
                else:
                    output[j][0] = 1
            '''
            loss = self.compute_loss(output, y)
            
            if self.record:
                writer.add_scalar("Loss/train", loss, i)
            if i % 5000 == 0:
                print(f"epoch {i} loss : {loss}")
            self.backward(y)