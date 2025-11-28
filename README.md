## GANs Project: Face Generation
# GAN Face Generation – README

## Overview
This project implements a Generative Adversarial Network (GAN) to generate 64×64 face images using the CelebA dataset. The GAN consists of two models— a **Generator** and a **Discriminator**—that train together in an adversarial setup:
- The **Generator** creates fake face images.
- The **Discriminator** learns to classify images as real or fake.

As training progresses, the generator becomes better at producing realistic-looking faces.

---

## Project Structure

### Generator
- Takes a 128-dimensional latent vector as input.
- Uses a fully connected layer to create a 4×4×512 feature map.
- Upsamples via ConvTranspose2D layers (4×4 → 8×8 → 16×16 → 32×32 → 64×64).
- Final activation: **Tanh**, matching normalized image range.

### Discriminator
- Accepts a 3×64×64 image as input.
- Downsamples using convolutional layers.
- Outputs a single **(1, 1, 1)** score representing the probability of the image being real.
- Uses **LeakyReLU** activations for stability.

### Training Loop
- Alternates between discriminator and generator training steps.
- Logs losses for both networks.
- After each epoch, generates sample images using a fixed noise vector.

---

## Dataset
The project uses the **CelebA** dataset, a large-scale dataset of celebrity face images.

Important notes:
- The dataset is **biased**, containing mostly young, light-skinned Western celebrities.
- Generated faces reflect this bias and show limited demographic diversity.
- Images are resized and normalized to the range **[-1, 1]**.

---

## Model Design Choices
- A **DCGAN-style architecture** was chosen for stability and simplicity.
- **latent_dim = 128**, balancing expressive capacity with training stability.
- **Adam optimizer** (`betas=(0.5, 0.999)`) for smoother GAN convergence.
- **Standard GAN loss**, which aligns with course requirements and is easy to debug.
- **20 training epochs**, selected due to time and compute constraints.

These choices allow the model to learn the basic structure of faces while keeping training computationally manageable.

---

## Results
- Generated samples show recognizable face structure (eyes, nose, mouth positions).
- Images remain somewhat blurry and lack fine detail.
- Some mode collapse is visible (similar-looking outputs).
- The demographic bias of the dataset is clearly reflected in the generated samples.

---

## Potential Improvements
To achieve higher realism and diversity, the following improvements could be made:
- Use **WGAN-GP** or **LSGAN** for more stable training.
- Train for more epochs (100–200+).
- Increase model depth and capacity.
- Use a more balanced dataset (e.g., **FairFace**).
- Apply data augmentation.
- Introduce learning rate scheduling or adjust batch sizes.

---

## How to Run
1. Install dependencies (PyTorch, torchvision, numpy, matplotlib).
2. Download and preprocess the CelebA dataset.
3. Initialize the generator and discriminator.
4. Run the training loop.
5. View generated images after each training epoch.

---

## Files
- `face_generation.ipynb` – Main notebook
- `models.py` – Generator & Discriminator implementations
- `utils.py` – Helper functions
- `README.md` – Project documentation (this file)
