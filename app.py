import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class IntuitionNN(nn.Module):
    def __init__(self, input_size, layer_sizes, intuition_size):
        super(IntuitionNN, self).__init__()
        self.layers = nn.ModuleList()
        self.initial_weights = []
        self.intuition_layer = nn.Linear(input_size, intuition_size)
        self.intuition_coefficients = torch.zeros(intuition_size)

        for i in range(len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i-1] if i > 0 else input_size, layer_sizes[i])
            self.layers.append(layer)
            self.initial_weights.append(layer.weight.clone().detach())
            if i > 1:
                extra_input_layer = nn.Linear(layer_sizes[i-2], layer_sizes[i])
                self.add_module(f'extra_input_layer_{i}', extra_input_layer)

    def forward(self, x, iteration):
        x_prev_prev = None
        intuition_output = self.intuition_layer(x) * self.intuition_coefficients
        for i, layer in enumerate(self.layers):
            if i > 1 and x_prev_prev is not None:
                extra_input_output = F.relu(getattr(self, f'extra_input_layer_{i}')(x_prev_prev))
                pre_computed = extra_input_output / 2
                x = F.relu(layer(x)) + pre_computed
            else:
                x = F.relu(layer(x))
            x_prev_prev = x_prev if i > 0 else x
            x_prev = x
        return x, intuition_output

    def compare_and_adjust(self, outputs, intuition_output):
        for i, layer in enumerate(self.layers):
            initial_weight = self.initial_weights[i]
            current_weight = layer.weight
            difference = current_weight - initial_weight
            adjustment = self.learn_from_difference(difference)
            layer.weight.data += adjustment
        
        # Update intuition coefficients based on outputs and intuition output
        intuition_adjustment = self.learn_from_intuition(outputs, intuition_output)
        self.intuition_coefficients += intuition_adjustment

    def learn_from_difference(self, difference):
        adjustment = difference * 0.0001  # Adjustment factor
        return adjustment
    
    def learn_from_intuition(self, outputs, intuition_output):
        intuition_difference = (outputs - intuition_output).pow(2).mean()
        intuition_adjustment = intuition_difference * 0.0001  # Adjustment factor for intuition
        return intuition_adjustment

def train_model(model, train_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # Learning rate
    all_losses = []
    all_weight_differences = {i: [] for i in range(len(model.layers))}
    all_intuition_differences = []

    for epoch in range(num_epochs):
        epoch_losses = []
        print(f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs if using images
            optimizer.zero_grad()
            outputs, intuition_output = model(inputs, epoch)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            epoch_losses.append(loss.item())
            model.compare_and_adjust(outputs, intuition_output)

            for i, layer in enumerate(model.layers):
                initial_weight = model.initial_weights[i]
                current_weight = layer.weight
                difference = (current_weight - initial_weight).pow(2).mean().item()
                all_weight_differences[i].append(difference)
            
            # Track intuition differences
            intuition_difference = (outputs - intuition_output).pow(2).mean().item()
            all_intuition_differences.append(intuition_difference)

        all_losses.append(np.mean(epoch_losses))
        print(f"Average Loss: {np.mean(epoch_losses)}")

    return all_losses, all_weight_differences, all_intuition_differences

def plot_results(all_losses, all_weight_differences, all_intuition_differences):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(all_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    for i in all_weight_differences:
        plt.plot(all_weight_differences[i], label=f'Layer {i} Weight Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Difference (MSE)')
    plt.title('Weight Differences over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(all_intuition_differences, label='Intuition Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Intuition Difference (MSE)')
    plt.title('Intuition Differences over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize model and train
input_size = 28*28  # MNIST images are 28x28 pixels
layer_sizes = [128, 64, 10]  # Example layer sizes
num_epochs = 10
intuition_size = 10  # Size of the intuition module

model = IntuitionNN(input_size=input_size, layer_sizes=layer_sizes, intuition_size=intuition_size)
all_losses, all_weight_differences, all_intuition_differences = train_model(model, train_loader, num_epochs)

# Plot the results
plot_results(all_losses, all_weight_differences, all_intuition_differences)
