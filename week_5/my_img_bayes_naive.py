import pickle
import numpy as np
from scipy.stats import multivariate_normal

file = "mnist_fashion.pkl"

with open(file, "rb") as datafile:
    x_train = pickle.load(datafile)
    y_train = pickle.load(datafile)
    x_test = pickle.load(datafile)
    y_test = pickle.load(datafile)


def compute_mean_var(x_train, y_train):

    x_mean = np.zeros((10, 784))
    # x_var = np.zeros((10, 784))
    x_co_var = np.zeros((10, 784, 784))
    for k in range(10):
        x_k_train = x_train[y_train == k]
        x_mean[k] = np.mean(x_k_train, axis=0)
        x_co_var[k] = np.cov(x_k_train, rowvar=False)

        # x_var[k] = np.var(x_k_train, axis=0) + 0.01
    return x_mean, x_co_var

# def inference(x_test, x_mean, x_var):
def inference(x_test, x_mean, x_cov):
    mle = np.zeros(10)
 
    for i in range(10):
        # log_term = 0
        # for j in range(784):
        #     term = -0.5* (np.log(2*np.pi) + np.log(x_var[i][j]) + (x_test[j]-x_mean[i][j])**2/x_var[i][j])
        #     log_term += term
        
        # # print(log_term)

        # mle[i] = log_term
        mle[i] = multivariate_normal.logpdf(
            x_test, mean=x_mean[i], cov=x_cov[i]
        )
    return np.argmax(mle)

     
        # # x_mean[k]

x_train = x_train.reshape(x_train.shape[0], 784)   
x_test = x_test.reshape(x_test.shape[0], 784) 
noise = np.random.normal(loc=0.0, scale=10.0, size=x_train.shape)
x_train_noisy = x_train + noise 

x_mean, x_var = compute_mean_var(x_train=x_train_noisy, y_train=y_train)
x_mean, x_cov = compute_mean_var(x_train=x_train_noisy, y_train=y_train)
correct = 0
answer = []
for i in range(len(x_test)):
    # pred = inference(x_test[i], x_mean, x_var)
    pred = inference(x_test[i], x_mean, x_cov)
    
    answer.append(pred)
    if pred == y_test[i]:
        correct += 1
np.savetxt("PRED_bayes.dat", answer, fmt='%d')
accuracy = correct/len(x_test)
print(accuracy)



