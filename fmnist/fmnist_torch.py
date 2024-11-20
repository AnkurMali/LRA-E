import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import datasets, transforms
import time

# Set random seed for reproducibility
torch.manual_seed(1234)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split train dataset into train and validation
train_data, val_data = torch.utils.data.random_split(train_dataset, [58000, 2000])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)



class MLP(nn.Module):
    def __init__(self, layer_sizes, device=None):
        """
        layer_sizes: list of int, sizes of each layer including input and output layers
        device: str or None, either 'cpu' or 'cuda'. If None, the device is automatically decided.
        """
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes
        #self.device = "cpu"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize weights and biases for each layer
        self.weights = nn.ModuleList()
        self.biases = nn.ModuleList()
        self.E = nn.ModuleList()  # Error propagation matrices
        
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False).to(self.device)
            self.weights.append(layer)
            if i > 0:
                # Error matrix for LRA update
                error_layer = nn.Linear(layer_sizes[i + 1], layer_sizes[i], bias=False).to(self.device)
                self.E.append(error_layer) 

        # Move model to the specified device
        print("Verifying weights in self.weights are on the correct device:")
        for i, layer in enumerate(self.weights):
              print(f"Layer {i}: {layer.weight.device}")
        self.to(self.device)
    
    def forward(self, X):
        """
        Forward pass to compute output
        X: Tensor, shape (batch_size, input_size)
        """
        X = X.to(self.device)
        activations = X
        activations = X.to(self.device)
        for i, layer in enumerate(self.weights):
            self.weights[i] = layer.to(self.device)
        for i in range(len(self.weights) - 1):
            pre_activation = self.weights[i](activations)
            activations = torch.tanh(pre_activation)
        
        logits = self.weights[-1](activations)
        self.z_last = F.softmax(logits, dim=1)
        return logits

    def compute_pre_activation(self, layer_index, X):
        """
        Compute pre-activation output for a specific layer
        layer_index: Index of the layer
        X: Input tensor
        """
        activations = X
        for i in range(layer_index):
            activations = torch.tanh(self.weights[i](activations))
        return self.weights[layer_index](activations)

    def compute_activation(self, layer_index, X):
        """
        Compute activation output for a specific layer
        layer_index: Index of the layer
        X: Input tensor
        """
        activations = X
        for i in range(layer_index + 1):
            activations = torch.tanh(self.weights[i](activations))
        return activations

    def compute_lra_updates(self, X_train, Y_train, optimizer, beta=0.1, gamma=1.0):
        """
        Compute the LRA updates and apply them using PyTorch.
        """
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)  # Ensure data is on the same device
        batch_size = X_train.size(0)

        # Forward pass
        logits = self.forward(X_train)
        e_last = self.z_last - Y_train

        # Initialize variables
        e = [None] * (len(self.layer_sizes)-1)
        d = [None] * (len(self.layer_sizes)-1)
        y_z = [None] * (len(self.layer_sizes)-1)
        dW = [None] * (len(self.layer_sizes)-1)
        dW_e = [None] * (len(self.layer_sizes)-2)

        # Backward error propagation
        e[-1] = e_last  # Final error
        for i in range(len(self.layer_sizes) - 2, 0, -1):  # From second-to-last layer to first hidden layer
            e[i] = e[i].to(self.E[i - 1].weight.device)
            d[i] = torch.matmul(e[i], self.E[i - 1].weight.T)  # Propagate error backward
            d_b = d[i] * beta  # Scale by beta
            pre_activation = self.compute_pre_activation(i - 1, X_train)
            d_b = d_b.to(pre_activation.device)

            # Align shapes
            if d_b.shape != pre_activation.shape:
                  print("Aligning d_b shape to pre_activation")
                  d_b = d_b.reshape_as(pre_activation)
            #y_z[i] = torch.tanh(self.compute_pre_activation(i - 1, X_train) - d_b)  # Adjust activation
            y_z[i] = torch.tanh(pre_activation - d_b)
            e[i - 1] = self.compute_activation(i - 1, X_train) - y_z[i]  # Compute new error

        # Compute gradients
        for i in range(len(self.layer_sizes) - 1):
            if i == 0:
                dW[i] = torch.matmul(e[i].T, X_train) / batch_size
            else:
                activation = self.compute_activation(i - 1, X_train)
                activation = activation.to(e[i].device)
                # Align shapes if necessary (debugging)
                if e[i].shape[0] != activation.shape[0]:
                  e[i] = e[i].reshape(activation.shape[0], -1)
                #dW[i] = torch.matmul(e[i].T, self.compute_activation(i - 1, X_train)) / batch_size
                dW[i] = torch.matmul(e[i].T, activation) / batch_size
            dW[i] = dW[i].T  # Transpose the gradient to match weight dimensions

        
        # Compute gradients for self.E
        for i in range(len(self.E)):
            activation = self.compute_activation(i, X_train)
            activation = activation.to(e[i+1].device)
            if e[i+1].shape[0] != activation.shape[0]:
                  e[i+1] = e[i+1].reshape(activation.shape[0], -1)
            dW_e[i] = torch.matmul(e[i+1].T, activation) / batch_size
            #dW_e[i] = torch.matmul(e[i + 1].T, self.compute_activation(i, X_train)) / batch_size    
       
        optimizer.zero_grad()  # Clear existing gradients
        for i, layer in enumerate(self.E):
                self.E[i] = layer.to(self.device)
        
        for i, layer in enumerate(self.weights):
            layer.weight.grad = dW[i].T.to(self.device)  # Assign gradients to weights
            
            
            if i < len(self.E):
                if dW_e[i].T.shape != self.E[i].weight.shape:
                    print(f"Shape mismatch detected. dW_e[{i}].T.shape: {dW_e[i].T.shape}, self.E[{i}].weight.shape: {self.E[i].weight.shape}")
                    # Reshape dW_e[i] to match self.E[i].weight
                    dW_e[i] = dW_e[i].reshape_as(self.E[i].weight.T)
            
                #print(f"self.E[{i}].weight.device: {self.E[i].weight.device}")
                self.E[i].weight.grad = dW_e[i].T.to(self.device)  # Assign gradients to error weights
        optimizer.step()  # Apply optimizer step


    

    def loss(self, y_pred, y_true):
        """
        Compute the loss
        y_pred: Predicted logits
        y_true: Ground truth (one-hot encoded)
        """
        y_true = y_true.to(y_pred.device)
        print(f"y_pred device: {y_pred.device}, y_pred shape: {y_pred.shape}")
        print(f"y_true device: {y_true.device}, y_true (argmax) shape: {y_true.argmax(dim=1).shape}")
        return F.cross_entropy(y_pred, y_true.argmax(dim=1))

# Hyperparameters
layer_sizes = [784, 256, 256, 256, 256, 10]
learning_rate = 0.001
num_epochs = 100

# Initialize model, optimizer, and loss function
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(layer_sizes).to(device)  # Change 'cpu' to 'cuda' if a GPU is available
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
start_time = time.time()
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        x_train, y_train = batch
        y_train_onehot = nn.functional.one_hot(y_train, num_classes=10).float()

        logits = model(x_train)
        y_train = y_train.to(logits.device)
        loss = criterion(logits, y_train)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        model.compute_lra_updates(x_train, y_train_onehot, optimizer)
        
        predictions = torch.argmax(logits, dim=1)
        correct_train += (predictions == y_train).sum().item()
        total_train += y_train.size(0)

    train_accuracy = correct_train / total_train * 100

    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for batch in val_loader:
            x_val, y_val = batch
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits = model(x_val)
            predictions = torch.argmax(logits, dim=1)
            predictions.to(device)
            y_val.to(device)
            correct_val += (predictions == y_val).sum().item()
            total_val += y_val.size(0)

    val_accuracy = correct_val / total_val * 100

    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Training Accuracy = {train_accuracy:.2f}%, Validation Accuracy = {val_accuracy:.2f}%")
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy

print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

# Test accuracy
correct_test = 0
total_test = 0

model.eval()
with torch.no_grad():
    for batch in test_loader:
        x_test, y_test = batch
        x_test, y_test = x_test.to(device), y_test.to(device)
        logits = model(x_test)
        predictions = torch.argmax(logits, dim=1)
        correct_test += (predictions == y_test).sum().item()
        total_test += y_test.size(0)

test_accuracy = correct_test / total_test * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

end_time = time.time()
print(f"Total Time Taken: {end_time - start_time:.2f} seconds")
