import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Learning rate
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
