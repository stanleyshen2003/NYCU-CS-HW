import numpy as np
from layers import nn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predict Result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

def evaluate(inputs, labels, model, text):
    pred = model.forward(inputs)
    loss = 0
    for j in range(pred.shape[0]):
        print(f"Iter{j} |    Ground truth: {labels[j]} |     prediction: {pred[j][0]}")
        loss += (pred[j][0] - labels[j])**2
        if pred[j][0] < 0.5:
            pred[j][0] = 0
        else:
            pred[j][0] = 1
    pred = np.round(pred)
    print(f"Loss={loss} accuracy={np.sum(pred == labels) / len(labels)}")

def draw_boudary(model, inputs, labels):
    # create a mesh to plot in
    x_min, x_max = 0,1
    y_min, y_max = 0,1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    # make the plot square
    ax.set_aspect('equal', adjustable='datalim')
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    for j in range(Z.shape[0]):
        if Z[j][0] < 0.5:
            Z[j][0] = 0
        else:
            Z[j][0] = 1
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # if the result is 0, it will be red, otherwise it will be blue
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    #also draw the training points
    for i in range(inputs.shape[0]):
        if labels[i] == 0:
            plt.plot(inputs[i][0], inputs[i][1], 'ro')
        else:
            plt.plot(inputs[i][0], inputs[i][1], 'bo')
    
    # turn off axis
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    hidden_units = 3
    lr = 0.1
    epochs = 30000
    activation = 'sigmoid'
    optimizer = 'SGD'

    inputs, labels = generate_linear()
    model = nn(inputs.shape[1], hidden_units, 1, lr=lr, epochs=epochs, activation=activation, optimizer=optimizer, record=False)
    model.train(inputs, labels, 'linear/lr' + str(lr) +'_hidden'+ str(hidden_units) + activation + optimizer)
    evaluate(inputs, labels, model, 'Linear')
    draw_boudary(model, inputs, labels)
    show_result(inputs, labels, model.forward(inputs))

    # inputs, labels = generate_XOR_easy()
    # model = nn(inputs.shape[1], hidden_units, 1, lr=lr, epochs=epochs, activation=activation, optimizer=optimizer, record=False)
    # model.train(inputs, labels, 'XOR/lr' + str(lr) +'_hidden'+ str(hidden_units) + activation + optimizer)
    # evaluate(inputs, labels, model, 'XOR')
    # draw_boudary(model, inputs, labels)
    # show_result(inputs, labels, model.forward(inputs))