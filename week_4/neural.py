
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

file = "mnist_fashion.pkl"

with open(file, "rb") as datafile:
    x_train = pickle.load(datafile)
    y_train = pickle.load(datafile)
    x_test = pickle.load(datafile)
    y_test = pickle.load(datafile)


print(x_train.shape)
print(x_test.shape[0])
print(y_test[0])

x_train = torch.from_numpy(x_train.reshape(60000, 784)).float()

x_test = torch.from_numpy(x_test.reshape(10000, 784)).float()

y_test = torch.from_numpy(y_test).long()
y_train = torch.from_numpy(y_train).long()



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() 
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)
      

    def forward(self, input):
        o_1 = F.relu(self.fc1(input))
        o_2 = F.relu(self.fc2(o_1))
        o_3 = F.relu(self.fc3(o_2))
        o_4 = F.relu(self.fc4(o_3))
        output = self.output(o_4)
        return output



model =  NeuralNetwork()

criterion = nn.CrossEntropyLoss()
 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)


for epoch in range(10):
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}')

model.eval()

with torch.no_grad():
    model_output = model(x_test)
    model_output = torch.argmax(model_output, dim=1)
    np.savetxt('PRED_mnist_fashion.dat', model_output, fmt='%d')
    correct = (model_output==y_test).sum()
    accuracy = correct/len(model_output)
    print(accuracy)


