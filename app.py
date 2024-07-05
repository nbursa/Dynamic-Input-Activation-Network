import torch
import numpy as np
from torchvision import datasets, transforms
from src.model import IntuitionNN, RegularNN
from src.train import train_model, train_regular_model, plot_comparison_results

if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    # Initialize and train IntuitionNN model
    input_size = 28*28  # MNIST images are 28x28 pixels
    layer_sizes = [128, 64, 10]  # Example layer sizes
    num_epochs = 10
    intuition_size = 10  # Size of the intuition module

    model_intuition = IntuitionNN(input_size=input_size, layer_sizes=layer_sizes, intuition_size=intuition_size)
    intuition_logs = train_model(model_intuition, train_loader, val_loader, num_epochs)

    # Initialize and train RegularNN model
    model_regular = RegularNN(input_size=input_size, layer_sizes=layer_sizes)
    regular_logs = train_regular_model(model_regular, train_loader, val_loader, num_epochs)

    # Plot the results
    plot_comparison_results(intuition_logs, regular_logs)
