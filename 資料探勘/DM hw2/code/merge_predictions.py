import csv
import pandas as pd
import numpy as np
def merge_predictions(files):
    predictions = []
    for (i, file) in enumerate(files):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for (j,row) in enumerate(reader):
                if i == 0:
                    temp = np.zeros(6)
                    temp[int(row[1])] = 1
                    predictions.append(temp)
                else:
                    predictions[j][int(row[1])] += 1
    return predictions

predictions = merge_predictions(['submission.csv', 'submission2.csv', 'submission3.csv', 'submission4.csv', 'submission5.csv'])

predictions = [np.argmax(prediction).item() for prediction in predictions]


with open('submission_new.csv', 'w') as f:
    f.write('index,rating\n')
    for i, answer in enumerate(predictions):
        answer = answer
        f.write('index_'+str(i)+','+str(answer)+'\n')