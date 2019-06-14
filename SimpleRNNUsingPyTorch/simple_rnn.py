import pdb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# datasets were numericalized using weka
train_data = pd.read_csv('C:/Development/DLNID/Datasets/NSL-KDD/KDDTrainRenameNominalValuesNormalized.csv', header=0, low_memory=False, dtype = np.float32)
test_data = pd.read_csv('C:/Development/DLNID/Datasets/NSL-KDD/KDDTestRenameNominalValuesNormalized.csv', header=0, low_memory=False, dtype = np.float32)

targets_pd = train_data.iloc[0:, 41]
features_pd = train_data.iloc[0:, 0:41]

test_targets_pd = test_data.iloc[0:, 41]
test_features_pd = test_data.iloc[0:, 0:41]

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.FloatTensor(features_pd.values)
# torch_tensor = torch.tensor(targets_df['targets'].values)
targetsTrain = torch.LongTensor(targets_pd.values)

featuresTest = torch.FloatTensor(test_features_pd.values)
# torch_tensor = torch.tensor(targets_df['targets'].values)
targetsTest = torch.LongTensor(test_targets_pd.values)

# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_pd) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


# Create RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, (h_n, h_c) = self.rnn(x, None)
        out = self.fc(out[:, -1, :]) 
        return out
    
# Create RNN
input_dim = 41    # input dimension
hidden_dim = 80  # hidden layer dimension; was 100 before
layer_dim =  1    # number of hidden layers; was 2 before
output_dim = 2   # output dimension

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

seq_dim = 1  
loss_list = []
iteration_list = []
accuracy_list = []
count = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # print("Training: ",i)
        train  = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels )
        # pdb.set_trace()
            
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 250 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                images = Variable(images.view(-1, seq_dim, input_dim))
                
                # Forward propagation
                outputs = model(images)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += labels.size(0)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data.item(), accuracy))

