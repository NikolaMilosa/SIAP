import torch
import torch.nn as nn
import pandas
from sklearn.model_selection import train_test_split
from args import get_args


# 1D convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=256, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=256, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # Permute tensor to match input shape expected by Conv1d layer
        x = x.permute(0, 2, 1)

        # Apply 1D convolution
        x = self.conv1(x)
        x = self.relu(x)

        # Apply GRU layer
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x


def convert_from_date_to_float(data):
    converted_data = pandas.to_datetime(data)
    return (converted_data - converted_data.min()).astype('timedelta64[s]').astype('int32').astype('float32')


def make_neural_network(data):
    # Convert date
    data['time'] = convert_from_date_to_float(data['time'])
    # Extract the input and output data
    x = data.drop(['PriceUSD'], axis=1).values.astype('float32')
    y = data['PriceUSD'].values.astype('float32')

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Convert the data to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize the network and define the loss function and optimizer
    net = Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    args = get_args()

    # Train the network
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        outputs = net(x_train.unsqueeze(1))
        loss = criterion(outputs, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print('Epoch {}, Loss: {}'.format(epoch, loss.item()))

    # Test the network on the testing set
    with torch.no_grad():
        test_outputs = net(x_test.unsqueeze(1))
        test_loss = criterion(test_outputs, y_test.unsqueeze(1))
        print('Test Loss: {}'.format(test_loss.item()))

    # Return the loss values as a dictionary
    return {
        'test_loss': test_loss.item()
    }
