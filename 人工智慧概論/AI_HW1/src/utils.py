import numpy as np

def evaluate(clf, dataset):
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0

    for x, y in dataset:
        #print(x)
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1

        prediction = clf.classify(x)
        if prediction == 1:
            if y == 1:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if y == 1:
                false_negatives += 1
            else:
                true_negatives += 1
    
    correct = true_positives + true_negatives
    print("False Positive Rate: %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("False Negative Rate: %d/%d (%f)" % (false_negatives, all_positives, false_negatives/all_positives))
    print("Accuracy: %d/%d (%f)" % (correct, len(dataset), correct/len(dataset)))

def integralImage(image):
    """
    Computes the integral image representation of a picture. 
    The integral image is defined as following:
    1. s(x, y) = s(x, y-1) + i(x, y), s(x, -1) = 0
    2. ii(x, y) = ii(x-1, y) + s(x, y), ii(-1, y) = 0
    Where s(x, y) is a cumulative row-sum, ii(x, y) is the integral image,
    and i(x, y) is the original image.
    The integral image is the sum of all pixels above and left of the current pixel
      Parameters:
        image : A numpy array with shape (m, n).
      Return:
        ii: A numpy array with shape (m, n) representing the integral image.
    """
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii