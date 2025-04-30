# Importing libraries

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchsummary import summary
from sklearn.model_selection import train_test_split
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading the data
df = pd.read_csv('../Work/riceClassification/riceClassification.csv')

df.drop('id', axis=1,inplace=True)

for col in df.columns:
    df[col] = df[col]/df[col].abs().max()

x = np.array(df.iloc[:,:-1])
y = np.array(df.iloc[:,-1])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3)

x_test,x_val,y_test,y_val = train_test_split(x_test,y_test,test_size=.5)


# Creating a class for the dataset
class dataset(Dataset):
    def __init__(self, x ,y):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# Appling the dataset class to the data
train_data = dataset(x_train,y_train)
val_data = dataset(x_val,y_val)
test_data = dataset(x_test,y_test)

# Loading the data with pytorch DataLoader
load_train = DataLoader(train_data, batch_size=8, shuffle=True)
load_val = DataLoader(val_data, batch_size=8, shuffle=True)
load_test = DataLoader(test_data, batch_size=8, shuffle=True)

# Building the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(x.shape[1], 10)
        self.linear = nn.Linear(10,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.input_layer(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

model = MyModel().to(device)

# Modle Training
total_train_loss_plot = []
total_train_acc_plot = []
total_val_loss_plot = []
total_val_acc_plot = []

epochs = 10
loss = nn.BCELoss()
opt = Adam(model.parameters(), lr = 1e-3)

for epoch in range(epochs):
    total_train_acc = 0
    total_train_loss = 0
    total_val_acc = 0
    total_val_loss = 0

    for data in load_train:
        inputs, label = data
        pred = model(inputs).squeeze(1)
        batch_loss = loss(pred,label)
        total_train_loss += batch_loss.item()
        total_train_acc += (pred.round() == label).sum().item()

        # Back propagation
        batch_loss.backward()
        opt.step()
        opt.zero_grad()

    # trying the model on validation data
    with torch.inference_mode():
        for val_data in load_val:
            val_inputs, val_label = val_data
            val_pred = model(val_inputs).squeeze(1)
            val_batch_loss = loss(val_pred,val_label)
            total_val_loss += val_batch_loss.item()
            total_val_acc += (val_pred.round() == val_label).sum().item()
    total_train_loss_plot.append(round(total_train_loss / len(load_train.dataset),3))
    total_train_acc_plot.append(round(total_train_acc / len(load_train.dataset) * 100,3))
    total_val_loss_plot.append(round(total_val_loss / len(load_val.dataset),3))
    total_val_acc_plot.append(round(total_val_acc / len(load_val.dataset) * 100,3))
