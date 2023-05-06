import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# 1D convolutional neural network
from torch import optim


from src.constants import output_attribute

torch.manual_seed(42)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def create_neural_network(data, path, epoch_num):
    # Extract the input and output data
    x = data.drop([output_attribute], axis=1).values
    y = data[output_attribute].values.reshape(-1, 1)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


    # Instantiate neural network model
    model = Net()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(epoch_num):
        # Convert data to tensors
        inputs = torch.from_numpy(x_train).float()
        targets = torch.from_numpy(y_train).float()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epoch_num}], Loss: {loss.item():.4f}")

    # Evaluate the model on the test set
    with torch.no_grad():
        inputs = torch.from_numpy(x_test).float()
        targets = torch.from_numpy(y_test).float()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        print(f"Test Loss: {loss.item():.4f}")

    # Test the neural network
    with torch.no_grad():
        outputs = model(torch.Tensor(x_test))
        print("Predicted\tActual")
        for i in range(len(outputs)):
            if i % 15 == 0:
                print(f'{outputs[i].item()}\t{y_test[i]}')


