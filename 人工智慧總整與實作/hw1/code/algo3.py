import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from img2vec_pytorch import Img2Vec
from sklearn.metrics import accuracy_score,davies_bouldin_score
import os
import seaborn as sns
from scipy.stats import entropy


def get_entropy(labels):
    _,counts = np.unique(labels, return_counts=True)
    return entropy(counts)

# prepare data
img2vec = Img2Vec()

data_dir = './data/dataset'
datasize = 100
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

folders = [train_dir, test_dir]
classes = ['traffic_light', 'zebra_crossing', 'bus', 'double_yellow_line', 'nothing']
features = []

for category in classes:
    for folder in folders:
        for img_path in os.listdir(os.path.join(folder, category)):
            img_path_ = os.path.join(folder, category, img_path)
            img = Image.open(img_path_).convert(mode='RGB')
            img_features = img2vec.get_vec(img)
            features.append(img_features)

features = np.array(features)

# set & train model
np.random.seed(1)
kmeans = KMeans(n_clusters=len(classes),init='random')
kmeans.fit(features)
Z = kmeans.predict(features)
print(Z)

# evaluate
true_labels = []
for i in range(len(classes)):
    class_i = [sum([1 for num in Z[datasize*(i):datasize*(i+1)] if num == j]) for j in range(len(classes))]
    class_i = np.argmax(class_i)
    for j in range(datasize):
        true_labels.append(class_i)
true_labels = np.array(true_labels)

accuracy = round(accuracy_score(Z, true_labels), 4)
print(f"Accuracy using k-means clustering: {accuracy}")

davies = davies_bouldin_score(features, Z)
print(f"Davies Bouldin score using k-means clustering: {davies}")

total = 0
for i in range(len(classes)):
    indices = [int(j/datasize) for j, x in enumerate(Z) if x == i]
    total += get_entropy(indices)
    print(f"Entropy of cluster {i}: {get_entropy(indices)}")
print(f"Entropy sum of all clusters: {total}")