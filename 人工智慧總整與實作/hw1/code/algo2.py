import os
import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# prepare data

img2vec = Img2Vec()

data_dir = './data/dataset_aug'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

classes = ['traffic_light', 'zebra_crossing', 'bus', 'double_yellow_line', 'nothing']
data = {}
for j, dir_ in enumerate([train_dir, test_dir]):
    features = []
    labels = []
    for category in classes:
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_)

            img_features = img2vec.get_vec(img)

            features.append(img_features)
            labels.append(category)

    data[['training_data', 'testing_data'][j]] = features
    data[['training_labels', 'testing_labels'][j]] = labels

fold_n = 0      # 0 1 2 3
data['validation_data'] = data['training_data'][fold_n*20:(fold_n+1)*20]
data['validation_labels'] = data['training_labels'][fold_n*20:(fold_n+1)*20]
for i in range(20):
   del data['training_data'][fold_n*20]
   del data['training_labels'][fold_n*20]
# train model
model = RandomForestClassifier(random_state=0)
model.fit(data['training_data'], data['training_labels'])

# validation performance
y_pred = model.predict(data['training_data'])
score = accuracy_score(y_pred, data['training_labels'])
print(f'training accuracy: {score}')

# validation performance
y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred, data['validation_labels'])
print(f'validation accuracy: {score}')

# test performance
y_pred = model.predict(data['testing_data'])
score = accuracy_score(y_pred, data['testing_labels'])
print(f'testing accuracy: {score}')

# # save the model
# with open('./model.p', 'wb') as f:
#     pickle.dump(model, f)
#     f.close()