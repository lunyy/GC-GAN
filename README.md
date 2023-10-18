
# GC-GAN: Graph Convolutional Network based conditonal Generative Adversarial Network with Class-Aware Discriminator

## Overview
We introduced a novel GCN-based Conditional GAN model that generates representative and realistic synthetic FC, aiming to provide a more precise and robust MDD diagnosis. 
By integrating GCN architecture into our GAN model, our framework demonstrated the capability to accurately capture the complex topology characteristics and intricate patterns present in FC. 
Additionally, the class-aware discriminator further enhanced the quality and effectiveness of synthetic FC generation in GC-GAN framework.
Furthermore, we introduced a topology refinement technique that optimizes the graph structure using the augmented synthetic FC, thereby improving the performance of MDD diagnosis.

## Code list

- config.py
- load_data.py : load data
- model.py : gcn based model arichtecture
- gcgan_ae.py : gcn autoencoder for generator
- gcgan_baseline.py : gcn classifier for baseline classifier and discriminator
- gcgan_train.py : gc-gan model
- gcgan_dataaugmenation.py : data augmentation
- gcgan_topologyrefinement.py : topology refinement
- mrmr_topology.py : mrmr feature selection algorithm based topology creation
- graph_utils.py : graph model function
- utils.py : acc,sen,spec function
- functional.py : mish activation
- etc 

## Requirements

To run this project, you will need:

- Python 3.7 or higher
- torch 1.8.0+cu111
- torchvision 0.9.0+cu111
- torch-geometric 2.0.3
- torch-scatter 2.0.9
- torch-sparse 0.6.12
- numpy 1.22.2
- etc.

### Installation

1. Make sure you have Python 3.7 or higher installed.

2. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Hyperparameters
| Hyperparameters         | SSGAN               | WGAN-GP             | ACGAN               | GC-GAN(ours)        |
|------------------------|---------------------|---------------------|---------------------|---------------------|
| Training Epochs        | 1000                | 500                 | 1000                | 1000                |
| Batch size             | 100                 | 100                 | 100                 | 100                 |
| Optimizer              | Adam (0.5,0.9)       | Adam (0.5,0.9)       | Adam (0.5,0.9)       | Adam (0.5,0.9)       |
| Learning rate scheduler| ExponentialLR (0.998)| ExponentialLR (0.998)| ExponentialLR (0.998)| ExponentialLR (0.998)|
| Activation function    | Mish                | Mish                | Mish                | Mish                |
| Weight decay           | 5e-03               | 5e-03               | 5e-03               | 5e-03               |
| Discriminator Hidden layers | 112-112-224   | 112-112-224         | 112-112-224         | 112-112-224         |
| Discriminator Learning rate | 1e-04         | 9e-05               | 7e-05               | 9e-05               |
| Discriminator Loss function  | Unsupervised loss + Cross entropy loss | Wasserstein loss + $\lambda$*gradient penalty | Adversarial loss + Cross entropy loss | Cross entropy loss |
| Generator Hidden layers | 57-112-112       | 57-112-112           | 57-112-112           | 57-112-112           |
| Generator Learning rate | 1e-04           | 1e-04               | 1e-04               | 1e-04               |
| Generator Loss function | Feature matching loss | Wasserstein loss + Cross entropy loss | Adversarial loss + Cross entropy loss | Cross entropy loss + $\alpha$*Mean squared error loss |
| Classifier Hidden layers| -                 | 112-112-224         | -                   | -                   |
| Classifier Learning rate| -                 | 1e-04               | -                   | -                   |
| Classifier Loss function| -                 | Cross entropy loss  | -                   | -                   |
| Gradient Penalty ($\lambda$) | -          | 10                  | -                   | -                   |
| Decrease ratio of $\alpha$  | -             | -                   | -                   | 0.995               |

