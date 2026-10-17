# Federated Learning on MNIST Using PyTorch

## Overview

This project implements **Federated Learning from scratch using PyTorch**, without using federated learning frameworks such as Flower.

The goal is to understand how multiple clients can collaboratively train a global Convolutional Neural Network (CNN) while keeping their training data local.

The project uses the MNIST handwritten digit dataset and implements **Federated Averaging (FedAvg)** to aggregate locally trained model parameters.

## Objectives

- Understand the fundamentals of federated learning.
- Implement a client-server training workflow without a federated learning framework.
- Train a CNN locally on multiple clients.
- Implement weighted model aggregation using FedAvg.
- Evaluate the global model on the MNIST test dataset.

## Technologies Used

- Python
- PyTorch
- Torchvision
- MNIST dataset

## Model Architecture

The CNN consists of:

1. Convolutional layer with 32 output channels
2. ReLU activation and Batch Normalization
3. Convolutional layer with 16 output channels
4. ReLU activation and Batch Normalization
5. Flatten layer
6. Fully connected layer with 128 hidden units
7. ReLU activation
8. Output layer with 10 classes

The model uses Cross-Entropy Loss and the Adam optimizer.

## Dataset and Client Setup

The MNIST dataset contains 60,000 training images and 10,000 test images, each representing one of ten handwritten digits (0–9).

Three simulated clients are created using different label distributions:

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

### FedAvg

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

## Experimental Results

### Global Model

| Metric | Result |
|---|---:|
| Dataset | MNIST |
| Number of clients | 3 |
| Federated communication rounds | 10 |
| Global test accuracy | **98.49%** |

### Local Model Evaluation

| Client | Test accuracy |
|---|---:|
| Client 1 | 71.36% |
| Client 2 | 57.04% |
| Client 3 | 36.73% |

The local models are evaluated on digit classes excluded from their respective training datasets. These results measure performance on unseen classes and should not be directly compared with the global model's accuracy on the full test set.

## Key Observations

- The global model achieved 98.49% accuracy on the full MNIST test dataset.
- Local training allows clients to learn from their own data without directly sharing raw training examples.
- FedAvg combines local model updates into a shared global model.
- Differences between local and global evaluation results highlight the effect of client-specific data distributions.

## Limitations and Future Work

- Compare federated training with a centralized CNN baseline.
- Record global test accuracy and loss after every communication round.
- Investigate the effect of different client partitions and numbers of clients.
- Examine the effect of Batch Normalization statistics during aggregation.
- Explore disjoint and more strongly non-IID data partitions.
- Extend the implementation to more complex datasets and models.

## Privacy Note

This implementation keeps training examples local in the simulated client workflow, but it does not implement differential privacy, secure aggregation, or cryptographic protection of model updates. Federated learning alone does not guarantee privacy.

## How to Run

1. Install Python and PyTorch.
2. Install Torchvision:

   ```bash
   pip install torch torchvision
   ```

3. Run the Python training script.
4. The MNIST dataset will be downloaded automatically if it is not already available.
5. The final global model can be saved using PyTorch's `state_dict` mechanism.

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
