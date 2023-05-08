import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.constants import output_attribute


# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Create a custom dataset class for easier data handling
class EthereumDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_neural_network(data, path, epoch_num):
    # Split the data into 70% training and 30% testing
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # Separate the input features and output (PriceUSD)
    X_train = train_data.drop(output_attribute, axis=1).values
    y_train = train_data[output_attribute].values
    X_test = test_data.drop(output_attribute, axis=1).values
    y_test = test_data[output_attribute].values

    # Create the datasets and data loaders
    train_dataset = EthereumDataset(X_train, y_train)
    test_dataset = EthereumDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    # Set the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the MLP model and define the loss function, optimizer, and learning rate
    input_size = 8
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 1
    learning_rate = 0.001

    model = MLP(input_size, hidden_size1, hidden_size2, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    num_epochs = epoch_num
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        # Print the loss for this epoch
        if epoch % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

    # Evaluate the model on the test set
    model.eval()

    with torch.no_grad():
        y_pred = []
        y_true = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            y_pred.extend(outputs.squeeze().tolist())
            y_true.extend(targets.tolist())

    # Calculate R-squared and RMSE
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    print(f"R-squared: {r2:.6f}")
    print(f"RMSE: {rmse:.6f}")

    # Test the neural network
    with torch.no_grad():
        outputs = model(torch.Tensor(X_test))
        print("Predicted\tActual")
        for i in range(len(outputs)):
            if i % 15 == 0:
                print(f'{outputs[i].item()}\t{y_test[i]}')


