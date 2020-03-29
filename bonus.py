import argparse
import pandas as pd
import torch
from torch import nn
from torch import optim
import random
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from torch.autograd import Variable
import matplotlib.pyplot as plt



# Whatever other imports you need

# You can implement classes and helper functions here too.
class Net(nn.Module):

    def __init__(self, input_size, hidden_size, activation):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        if self.activation:
            self.nonlinear = activation[self.activation]
        
        if self.hidden_size >0:
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, 1)
        else:
            self.fc1 = nn.Linear(self.input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        if activation:
            x = self.nonlinear(x)
        if hidden_size != 0:
            x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def sample_data(batch_size, df):
    choices = []
    sample= []
    for i in range(batch_size):
        author1 = random.choice(df.author)
        author2 = random.choice(df.author)
        while author1 == author2:
            author2 = random.choice(df.author)
        
        seed_entries_1 = (df[df["author"] == author1]).sample(n=2, replace=True)
        seed_entries_2 = (df[df["author"] == author2]).sample(n=2, replace=True)
        if random.random() < 0.5:
            choices.append((seed_entries_1.iloc[0], seed_entries_1.iloc[1], 0))
        else:
            choices.append((seed_entries_1.iloc[0], seed_entries_2.iloc[0], 1))

    for choose in choices:
        author1_data = list(choose[0])[2:]
        author2_data = list(choose[1])[2:]
        combine = Variable(torch.Tensor((author1_data + author2_data)))
        index = torch.Tensor([choose[2]])
        sample.append((combine, index))
    return sample


def pred(inputs):
    if net(inputs) > 0.5:
        prediction = 1
    else:
        prediction = 0
    return prediction
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    parser.add_argument("--hidden", dest="hidden", type=int, default="0", help="The size of the hidden layer")
    parser.add_argument("--activation", dest="activation", type=str, default=None, help="Name of the non-linearity")
    parser.add_argument("--epoch", dest="epoch", type=int, default="100", help="epoch size.")
    parser.add_argument("--batch", dest="batch", type=int, default="20", help="epoch size.")
    # and any other options you may think you want/need.  Document
    # everything.

    args = parser.parse_args()
    
    # map of functions
    non_linearities = {"sigmoid": nn.Sigmoid, "tahn": nn.Tanh, "relu": nn.ReLU}
    if args.activation:
        activation = non_linearities[args.activation]
    else:
        activation = None
    
    # read data
    print("Reading {}...".format(args.featurefile))
    data = pd.read_csv(args.featurefile)
    train = data[data.train_test == 'train']
    train_X = train.iloc[:, 1:]
    
    test = data[data.train_test == 'test']
    test_X = test.iloc[:, 1:]
    test_X = test_X.reset_index(drop=True)
    
    input_size = len(train_X.columns) - 2
    hidden_sizes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    batch_size = args.batch
    recalls = []
    precisions = []
    
    for hidden_size in hidden_sizes:
        net = Net(input_size*2, hidden_size, activation)
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.BCELoss()
    
        sample = sample_data(batch_size, train_X)
        for e in range(args.epoch):
            for inputs, label in sample:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

        sample_test = sample_data(batch_size, test_X)
        test_data = [inputs for inputs, label in sample_test]
        labels = [label for inputs, label in sample_test]
    
        prediction = []
        for inputs in test_data:
            preds = pred(inputs)
            prediction.append(preds)
        recalls.append(recall_score(labels,prediction))
        precisions.append(precision_score(labels,prediction))

    plot_data = pd.DataFrame({"hiddensize": hidden_sizes, "recall": recalls, "pred": precisions})
    plt.plot( 'hiddensize', 'recall', data=plot_data, color='blue', label="recalls")
    plt.plot( 'hiddensize', 'pred', data=plot_data, color='red', label="precisions")
    plt.legend(('Recalls', 'Precisions'),
           loc='upper right')
    plt.show()
