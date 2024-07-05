import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

def train_model(model, train_loader, val_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    all_losses = []
    all_val_losses = []
    all_accuracies = []
    all_val_accuracies = []
    all_weight_differences = {i: [] for i in range(len(model.layers))}
    all_intuition_differences = []
    all_lr = []
    all_grad_norms = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        correct = 0
        total = 0
        epoch_start_time = time.time()
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

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Log gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            all_grad_norms.append(total_norm)

            for i, layer in enumerate(model.layers):
                initial_weight = model.initial_weights[i]
                current_weight = layer.weight
                difference = (current_weight - initial_weight).pow(2).mean().item()
                all_weight_differences[i].append(difference)
            
            # Track intuition differences
            intuition_difference = (outputs - intuition_output).pow(2).mean().item()
            all_intuition_differences.append(intuition_difference)

        all_losses.append(np.mean(epoch_losses))
        all_accuracies.append(100 * correct / total)
        all_lr.append(optimizer.param_groups[0]['lr'])
        epoch_duration = time.time() - epoch_start_time
        print(f"Average Loss: {np.mean(epoch_losses)}, Accuracy: {100 * correct / total}%, Duration: {epoch_duration}s")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs if using images
                outputs, _ = model(inputs, epoch)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        all_val_losses.append(val_loss)
        all_val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")

        # Step the scheduler
        scheduler.step()

    return {
        'train_losses': all_losses,
        'val_losses': all_val_losses,
        'train_accuracies': all_accuracies,
        'val_accuracies': all_val_accuracies,
        'weight_differences': all_weight_differences,
        'intuition_differences': all_intuition_differences,
        'learning_rates': all_lr,
        'grad_norms': all_grad_norms
    }

def train_regular_model(model, train_loader, val_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    all_losses = []
    all_val_losses = []
    all_accuracies = []
    all_val_accuracies = []
    all_lr = []
    all_grad_norms = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        correct = 0
        total = 0
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")

        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs if using images
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Log gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            all_grad_norms.append(total_norm)

        all_losses.append(np.mean(epoch_losses))
        all_accuracies.append(100 * correct / total)
        all_lr.append(optimizer.param_groups[0]['lr'])
        epoch_duration = time.time() - epoch_start_time
        print(f"Average Loss: {np.mean(epoch_losses)}, Accuracy: {100 * correct / total}%, Duration: {epoch_duration}s")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs if using images
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        all_val_losses.append(val_loss)
        all_val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")

        # Step the scheduler
        scheduler.step()

    return {
        'train_losses': all_losses,
        'val_losses': all_val_losses,
        'train_accuracies': all_accuracies,
        'val_accuracies': all_val_accuracies,
        'learning_rates': all_lr,
        'grad_norms': all_grad_norms
    }

def plot_comparison_results(intuition_logs, regular_logs):
    plt.figure(figsize=(18, 12))

    # Training and Validation Loss
    plt.subplot(2, 3, 1)
    plt.plot(intuition_logs['train_losses'], label='IntuitionNN Train Loss', color='blue')
    plt.plot(intuition_logs['val_losses'], label='IntuitionNN Val Loss', color='lightblue')
    plt.plot(regular_logs['train_losses'], label='RegularNN Train Loss', color='orange')
    plt.plot(regular_logs['val_losses'], label='RegularNN Val Loss', color='lightcoral')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()

    # Training and Validation Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(intuition_logs['train_accuracies'], label='IntuitionNN Train Accuracy', color='blue')
    plt.plot(intuition_logs['val_accuracies'], label='IntuitionNN Val Accuracy', color='lightblue')
    plt.plot(regular_logs['train_accuracies'], label='RegularNN Train Accuracy', color='orange')
    plt.plot(regular_logs['val_accuracies'], label='RegularNN Val Accuracy', color='lightcoral')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()

    # Learning Rate
    plt.subplot(2, 3, 3)
    plt.plot(intuition_logs['learning_rates'], label='IntuitionNN Learning Rate', color='blue')
    plt.plot(regular_logs['learning_rates'], label='RegularNN Learning Rate', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.legend()

    # Gradient Norms
    plt.subplot(2, 3, 4)
    plt.plot(intuition_logs['grad_norms'], label='IntuitionNN Gradient Norms', color='blue')
    plt.plot(regular_logs['grad_norms'], label='RegularNN Gradient Norms', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms over Iterations')
    plt.legend()

    # Weight Differences for IntuitionNN
    plt.subplot(2, 3, 5)
    for i in intuition_logs['weight_differences']:
        plt.plot(intuition_logs['weight_differences'][i], label=f'IntuitionNN Layer {i} Weight Difference', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Weight Difference (MSE)')
    plt.title('Weight Differences over Iterations (IntuitionNN)')
    plt.legend()

    # Intuition Differences for IntuitionNN
    plt.subplot(2, 3, 6)
    plt.plot(intuition_logs['intuition_differences'], label='Intuition Difference', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Intuition Difference (MSE)')
    plt.title('Intuition Differences over Iterations (IntuitionNN)')
    plt.legend()

    plt.tight_layout()
    plt.show()
