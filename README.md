# Federated Learning on MNIST Using PyTorch

## Overview

This project implements **Federated Learning from scratch using PyTorch**, without using federated learning frameworks such as Flower.

The goal is to understand how multiple clients can collaboratively train a global Convolutional Neural Network (CNN) while keeping their training data local.

The project uses the MNIST handwritten digit dataset and implements **Federated Averaging (FedAvg)** to aggregate locally trained model parameters. A centralized CNN is also trained as a baseline for comparison.

## Objectives

- Understand the fundamentals of federated learning.
- Implement a client-server training workflow without a federated learning framework.
- Train CNN models locally on multiple clients.
- Implement weighted model aggregation using FedAvg.
- Compare federated learning against centralized training.
- Evaluate model performance on the MNIST test dataset.

## Technologies Used

- Python
- PyTorch
- Torchvision
- MNIST dataset

## Model Architecture

The CNN consists of:

1. Convolutional layer with 32 output channels.
2. ReLU activation and Batch Normalization.
3. Convolutional layer with 16 output channels.
4. ReLU activation and Batch Normalization.
5. Flatten layer.
6. Fully connected layer with 128 hidden units.
7. ReLU activation.
8. Output layer with 10 classes.

The model uses Cross-Entropy Loss and the Adam optimizer with a learning rate of 0.001.

## Dataset and Client Setup

The MNIST dataset contains 60,000 training images and 10,000 test images, representing ten handwritten digit classes (0–9).

Three simulated clients are created using different label distributions.

| Client | Excluded training labels | Test labels |
|---|---|---|
| Client 1 | 0, 1, 3 | 0, 1, 3 |
| Client 2 | 2, 5, 6 | 2, 5, 6 |
| Client 3 | 7, 8, 9 | 7, 8, 9 |

Each client trains on the remaining seven digit classes.

**Note:** The clients' training datasets overlap. This is a simulated non-IID label-distribution experiment, not a disjoint partition of the dataset.

## Federated Learning Workflow

The project follows these steps:

1. Initialize a global CNN model.
2. Distribute the global model weights to all clients.
3. Train each client model locally on its own dataset.
4. Collect the locally updated model parameters.
5. Aggregate the parameters using weighted Federated Averaging.
6. Update the global model with the aggregated parameters.
7. Repeat the process for 10 communication rounds.
8. Evaluate the global model on the complete MNIST test dataset.

### Federated Averaging (FedAvg)

The global model parameters are computed using a weighted average:

\[
w_{\text{global}} =
\sum_{k=1}^{K}\frac{n_k}{N}w_k
\]

where:

- \(K\) is the number of clients.
- \(n_k\) is the number of training examples at client \(k\).
- \(N\) is the sum of the client dataset sizes.
- \(w_k\) represents the locally trained model parameters.

## Centralized vs. Federated Results

The same CNN architecture was evaluated using centralized training and federated training.

| Experimental setting | Centralized CNN | Federated CNN (FedAvg) |
|---|---:|---:|
| Training setup | Entire training dataset | 3 simulated clients |
| Training epochs / communication rounds | 10 epochs | 10 rounds |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Test dataset | Full MNIST test set | Full MNIST test set |
| **Test accuracy** | **98.05%** | **98.49%** |
| Test error rate | 1.95% | 1.51% |

The federated model achieved **98.49% test accuracy**, compared with **98.05%** for the centralized CNN, a difference of 0.44 percentage points in this experiment.

These results are from individual runs. The observed difference does not establish that federated learning is inherently better than centralized training. Random initialization, data ordering, client data overlap, and Batch Normalization statistics may influence the results. Repeated experiments under controlled conditions are needed for a robust comparison.

## Local Client Evaluation

The local models were evaluated on test subsets containing the digit classes excluded from their respective training datasets.

| Client | Test accuracy |
|---|---:|
| Client 1 | 71.36% |
| Client 2 | 57.04% |
| Client 3 | 36.73% |

These measurements evaluate generalization to classes unseen by each local model. They should not be directly compared with the global model's accuracy on the complete MNIST test set.

## Key Observations

- The global FedAvg model achieved 98.49% accuracy on the full MNIST test dataset.
- The centralized CNN achieved 98.05% accuracy in the current run.
- Local models showed different levels of performance on their respective unseen-class test subsets.
- FedAvg combines local model updates into a shared global model.
- The experiment demonstrates a basic federated learning workflow without relying on a dedicated federated learning framework.

## Limitations and Future Work

- Repeat centralized and federated experiments with multiple random seeds.
- Compare mean accuracy and standard deviation across runs.
- Record global test loss and accuracy after every communication round.
- Investigate the effect of client count and different data partitions.
- Evaluate disjoint and more strongly non-IID client partitions.
- Examine the effect of Batch Normalization statistics during aggregation.
- Measure training time and communication overhead.
- Extend the implementation to more complex datasets and models.

## Model Saving

The final global model can be saved using PyTorch's `state_dict` mechanism:

```python
torch.save(global_model.state_dict(), "federated_mnist_cnn.pth")
```

## How to Run

1. Install Python and PyTorch.
2. Install the required dependencies:

   ```bash
   pip install torch torchvision
   ```

3. Run the Python training script.
4. The MNIST dataset will be downloaded automatically if it is not already available.
5. The training process will simulate three clients and perform 10 federated communication rounds.

## Privacy Note

This implementation keeps training examples local in the simulated client workflow, but it does not implement differential privacy, secure aggregation, or cryptographic protection of model updates. Federated learning alone does not guarantee privacy.

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
