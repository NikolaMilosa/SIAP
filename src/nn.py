import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

from src.constants import output_attribute


# Define the RNN architecture
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# Create a custom dataset class for easier data handling
class EthereumDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_neural_network(data, epoch_num):
    # Separate the input features and output (PriceUSD)
    X = data.drop(output_attribute, axis=1).values
    y = data[output_attribute].values

    # Create the datasets and data loaders
    dataset = EthereumDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Set the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the RNN model and define the loss function, optimizer, and learning rate
    input_size = 8
    hidden_size = 256
    num_layers = 2
    output_size = 1
    learning_rate = 0.001

    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    num_epochs = epoch_num
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))

            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        # Print the loss for this epoch
        if epoch % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

    # Evaluate the model on the dataset
    model.eval()

    with torch.no_grad():
        y_pred = []
        y_true = []
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.unsqueeze(1))

            y_pred.extend(outputs.squeeze().tolist())
            y_true.extend(targets.tolist())

        # Calculate R-squared and RMSE
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    print(f"R-squared: {r2:.6f}")
    print(f"RMSE: {rmse:.6f}")
