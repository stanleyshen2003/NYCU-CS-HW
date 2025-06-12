import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np


data = pd.read_csv('test.csv')
data.drop(columns=['date'], inplace=True)
columns = data.columns

data['ItemName'] = data['ItemName'].str.replace(' ', '')
print(data.head())
for column in columns[1:]:
    data[column].replace(['#', '*', 'x', 'A'], [None, None, None, None], inplace=True)
    data[column] = data[column].apply(pd.to_numeric, errors='coerce')
item_amount = data.groupby('ItemName').groups.keys().__len__()
data = data.to_numpy()

row_n, col_n = data.shape
item_names = data[0:item_amount, 0].reshape(-1)
dnew = {}
mean = {}
for name in item_names:
    dnew[name] = []
    

# insert into dictionary
for i in range(0, row_n, item_amount):
    for j in range(0, item_amount):
        dnew[item_names[j]].append(data[i + j, 1:].tolist())


for key in dnew.keys():
    dnew[key] = np.array(dnew[key]).flatten()
    mean[key] = np.nanmean(dnew[key])
    for i in range(1,dnew[key].shape[0]-1):
        if np.isnan(dnew[key][i]):
            if not np.isnan(dnew[key][i - 1]) and not np.isnan(dnew[key][i + 1]):
                dnew[key][i] = (dnew[key][i- 1] + dnew[key][i+ 1]) / 2

# count the number of nan values in each column
count = np.zeros(item_amount)
for i, key in enumerate(dnew.keys()):
    for j in range(dnew[key].shape[0]):
        if np.isnan(dnew[key][j]):
            count[i] += 1
            dnew[key][j] = mean[key]
print('number of nans after filling:')
for i in range(item_amount):
    print(f'{item_names[i]}: {count[i]}', end = ', ')
    
data = pd.DataFrame.from_dict(dnew)
data = data.apply(pd.to_numeric, errors='coerce')

# get the nn input data form and compute correlation
input_hours = 9
daily_data = data.to_numpy()
print(daily_data.shape)
draw_data = []
for i in range(0, daily_data.shape[0], input_hours):
    temp = daily_data[i:i+input_hours,:].transpose().flatten()
    temp = temp.tolist()
    #temp.append(daily_data[i + input_hours, 9])
    draw_data.append(temp)
print(len(draw_data))
labels = [[item+str(i) for i in range(input_hours)] for item in item_names]
labels = np.array(labels).flatten().tolist()
data = pd.DataFrame(draw_data, columns=labels)
print(data.size)
    
# threshold = 0.35
# element amount = 24*240
input_hours = 9
item_names = ['CO', 'PM10', 'PM2.5']# ['CO', 'NMHC', 'NO2', 'NOx', 'PM10', 'PM2.5', 'THC']
filter_item = []
for item in item_names:
    if item == 'PM2.5' or item == 'PM10':
        for i in range(4, input_hours):
            filter_item.append(item + str(i))
    else:
        for i in range(6, input_hours):
            filter_item.append(item + str(i))
new_data = data[filter_item]
print(new_data.size)
new_data = new_data.dropna()
print(new_data.head())
dataset = new_data.to_numpy()
dataset = dataset.astype(np.float64)
row_n, col_n = dataset.shape

print(dataset.shape)


class Linear():
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)
        self.intercept = np.random.randn(output_size)
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.intercept
    
    def backward(self, gradient, lr):
        gradient_out = np.dot(gradient, self.W.T)
        self.W -= lr * np.dot(self.input.T, gradient)
        self.intercept -= lr * np.sum(gradient, axis=0)
        return gradient_out

class Adagrad():
    def __init__(self, lr=0.01):
        self.lr = lr
        self.G = None
    def getlr(self, gradient):
        if self.G is None:
            self.G = 1
        self.G += np.mean(gradient**2)
        return self.lr / np.sqrt(self.G + 1e-7)
    
model = Linear(dataset.shape[1], 1)
loaded = np.load('closed_form_weights.npz')
model.W = loaded['weight']
model.intercept = loaded['intercept']

# inference
predict = model.forward(dataset)
predict = predict.flatten()
print(predict.shape)
#predict = (dataset[:,21] + dataset[:, 20]) / 2


predict = predict.tolist()
out = [['index_'+str(i), predict[i]] for i in range(len(predict))]
df = pd.DataFrame(out, columns=['index', 'answer'])
df.to_csv('predict.csv', index=False)



        