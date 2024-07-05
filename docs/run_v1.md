# Dynamic-Input-Activation-Network: Results Summary

## Results

This document summarizes the results of training the IntuitionNN and RegularNN models over 10 epochs. The metrics recorded include average loss, accuracy, duration, validation loss, and validation accuracy.

## Dataset

- **MNIST Dataset**: Both models were trained and evaluated on the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image is 28x28 pixels in grayscale.

## Model Descriptions

- **IntuitionNN Model**: A novel neural network architecture with dynamic input activation and an intuition module designed to enhance learning and adaptability.
- **RegularNN Model**: A standard feedforward neural network (multilayer perceptron) used as a baseline for comparison.

### Intuition NN Model

| Epoch | Average Loss | Accuracy (%) | Duration (s) | Validation Loss | Validation Accuracy (%) |
| ----- | ------------ | ------------ | ------------ | --------------- | ----------------------- |
| 1     | 1.9914       | 47.81        | 3.2879       | 1.6205          | 63.42                   |
| 2     | 1.3009       | 71.84        | 3.1722       | 1.0121          | 79.06                   |
| 3     | 0.8569       | 80.64        | 3.2545       | 0.7033          | 84.01                   |
| 4     | 0.6351       | 84.41        | 3.1819       | 0.5486          | 86.05                   |
| 5     | 0.5192       | 86.15        | 3.0991       | 0.4626          | 87.42                   |
| 6     | 0.4651       | 86.86        | 3.1236       | 0.4364          | 87.49                   |
| 7     | 0.4453       | 86.96        | 3.1580       | 0.4232          | 87.49                   |
| 8     | 0.4372       | 87.03        | 3.1077       | 0.4218          | 87.44                   |
| 9     | 0.4407       | 87.02        | 3.2606       | 0.4318          | 87.47                   |
| 10    | 0.4572       | 87.02        | 3.1997       | 0.4534          | 87.34                   |

### Regular NN Model

| Epoch | Average Loss | Accuracy (%) | Duration (s) | Validation Loss | Validation Accuracy (%) |
| ----- | ------------ | ------------ | ------------ | --------------- | ----------------------- |
| 1     | 2.0740       | 38.47        | 2.8385       | 1.8211          | 50.41                   |
| 2     | 1.6705       | 51.58        | 2.7491       | 1.5276          | 54.11                   |
| 3     | 1.4651       | 54.16        | 2.7596       | 1.3840          | 55.31                   |
| 4     | 1.3584       | 55.18        | 2.7437       | 1.3023          | 56.14                   |
| 5     | 1.2937       | 55.71        | 2.8699       | 1.2499          | 56.42                   |
| 6     | 1.2673       | 55.87        | 2.7731       | 1.2454          | 56.68                   |
| 7     | 1.2627       | 55.92        | 2.8769       | 1.2416          | 56.63                   |
| 8     | 1.2590       | 55.94        | 2.7770       | 1.2376          | 56.66                   |
| 9     | 1.2553       | 55.98        | 2.7121       | 1.2337          | 56.74                   |
| 10    | 1.2515       | 56.02        | 2.7200       | 1.2299          | 56.74                   |

---
