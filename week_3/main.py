import pickle
import numpy as np

data_fname = 'mnist_fashion.pkl'

with open(data_fname, 'rb') as data_files:
    x_train = pickle.load(data_files)
    y_train = pickle.load(data_files)
    x_test = pickle.load(data_files)
    y_test = pickle.load(data_files)

def my_1nn(x_train, y_train, x_test):
    x_train = x_train.reshape(60000, -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
    answer = []
    
    for t_image in x_test:
        min_distance = float('inf')
        best_label = None
        
        for x_image, y_label in zip(x_train, y_train):
            # Fixed the typo: e_distane -> distance
            # Also removed sqrt since we only need to compare distances
            distance = np.sum((x_image - t_image) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                best_label = y_label
        
        answer.append(best_label)
    
    return np.array(answer)

y_pred = my_1nn(x_train=x_train, y_train=y_train, x_test=x_test)

np.savetxt('PRED_mnist_fashion.dat', y_pred, fmt='%d')

def calculate_accuracy(pred, gt):
    correct = (pred == gt).sum()
    total = len(gt)
    accuracy = correct / total
    return accuracy

accuracy = calculate_accuracy(y_pred, y_test)
print(f"Your accuracy: {accuracy}")
print(f"Reference accuracy: 0.8497")
print(f"Difference: {abs(accuracy - 0.8497)}")