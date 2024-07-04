import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from src.model import IntuitionNN
from src.train import train_model, plot_results

if __name__ == "__main__":
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
