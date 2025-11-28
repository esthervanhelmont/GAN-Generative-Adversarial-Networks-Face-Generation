## GANs Project: Face Generation
GAN Face Generation – README
Overview

This project implements a Generative Adversial Network (GAN) to generate 64×64 face images using the CelebA dataset. The GAN consists of two models—a Generator and a Discriminator—that train together in an adversarial setup:

The Generator creates fake face images.

The Discriminator learns to classify images as real or fake.

As training progresses, the generator becomes better at producing realistic-looking faces.

Project Structure
Generator

Starts from a 128-dimensional latent vector.

Uses a fully connected layer to create a 4×4×512 feature map.

Upsamples via ConvTranspose2D layers: 4×4 → 8×8 → 16×16 → 32×32 → 64×64.

Final activation: Tanh (to match normalized image range).

Discriminator

Takes a 3×64×64 image as input.

Applies convolutional downsampling.

Outputs a single (1, 1, 1) score representing real/fake likelihood.

Uses LeakyReLU for stability.

Training Loop

Alternates between a discriminator step and a generator step.

Logs losses for both networks.

After every epoch, generates sample images using a fixed noise vector.

Dataset

The project uses the CelebA dataset, a large collection of celebrity face images.
Important considerations:

The dataset is biased, mainly consisting of young, light-skinned Western celebrity faces.

Generated samples reflect this bias and lack demographic diversity.

Images are resized and normalized to the range [-1, 1].

Model Design Choices

Chosen architecture: DCGAN-style, because it provides a simple, stable baseline for 64×64 image generation.

latent_dim = 128, balancing expressive power with training stability.

Adam optimizer with betas=(0.5, 0.999) to improve GAN convergence.

Standard GAN loss, as required for the course and simple to debug.

20 epochs due to compute and time limitations.

These choices allow the model to learn broad face structures while keeping training manageable.

Results

Generated faces show recognizable structure (eyes, nose, mouth placement).

Images remain somewhat blurry with limited detail.

Some mode collapse appears (similar-looking samples).

Outputs clearly reflect the dataset’s demographic bias.

Potential Improvements

To enhance realism, diversity, and stability, the model could be improved with:

WGAN-GP loss or LSGAN for stabler gradients.

More training epochs (100–200+).

A deeper generator and discriminator.

Dataset balancing or replacement with FairFace.

Data augmentation.

Learning rate scheduling or smaller batch sizes.

How to Run

Install dependencies (PyTorch, torchvision, numpy, matplotlib).

Download and preprocess the CelebA dataset.

Initialize the generator and discriminator.

Run the training loop.

View generated images after each epoch.

Files

face_generation.ipynb – Main notebook

models.py – Generator and Discriminator code

utils.py – Helper functions

README.md – Project description (this file)
