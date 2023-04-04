import torch
import torch.nn as nn
import json
from sklearn.model_selection import train_test_split

# 1D convolutional neural network
from torch.utils.data import DataLoader

from src.constants import output_attribute


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=256, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.gru = nn.GRU(input_size=256, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # Permute tensor to match input shape expected by Conv1d layer
        x = x.permute(0, 2, 1)

        # Apply 1D convolution
        x = self.conv1(x)
        x = self.sigmoid(x)

        # Apply GRU layer
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x


def create_neural_network(data, path, epoch_num):
    # Extract the input and output data
    x = data.drop([output_attribute], axis=1).values.astype('float32')
    y = data[output_attribute].values.astype('float32')

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Convert the data to PyTorch tensors and create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                                   torch.tensor(y_train, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                                  torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize the network and define the loss function and optimizer
    net = Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

    # Train the network
    for epoch in range(epoch_num):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.unsqueeze(1))
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Print the average loss every 10 epochs
        if epoch % 10 == 0:
            train_loss = running_loss / len(train_loader)
            print('Epoch {}, Training Loss: {}'.format(epoch, train_loss))


    # Test the network on the testing set
    with torch.no_grad():
        test_loss = 0.0
        for data in test_loader:
            inputs, labels = data
            test_outputs = net(inputs.unsqueeze(1))
            test_loss += criterion(test_outputs, labels.unsqueeze(1)).item()

        print('Test Loss: {}'.format(test_loss / len(test_loader)))

    # Serialize model and loss
    output_path = path.replace('input', 'output').replace('.csv', '.json')
    serialization_data = {
        'loss_history': 0
    }
    with open(output_path, 'w') as f:
        json.dump(serialization_data, f)
    torch.save(net.state_dict(), output_path.replace('output', 'output/model'))
