# Computer-Vision-Experiments
Task 1 — Graph Cut Segmentation (Assignment_1.ipynb)

Goal: Use a pretrained object detector to get bounding boxes for people in asm-1.jpg and asm-2.jpg, then run cv2.grabCut with those boxes. Compare iterations = {1, 3, 5}.

What to run:

Open Assignment_1.ipynb.

Provide or confirm a detector (notebooks include examples using a pretrained detector — modify if offline).

Run the GrabCut cells (the notebook will show original images, masks, foreground-only results, and overlays).

Check qualitative results and per-iteration differences (notebook computes simple metrics e.g., foreground pixel ratio; you can add IoU if ground-truth available).

Deliverables in repo: asm-1.jpg, asm-2.jpg and the notebook with visualizations.

Task 2 — FCN experiments (Assignment_2.ipynb & fcn_experiments/)

Goal: Implement FCN variants (32s/16s/8s) using a pretrained ResNet/VGG backbone, and compare upsampling via transpose convolution vs bilinear upsampling.

What’s included:

A small dataset or placeholder loader to run experiments quickly (10–20 images recommended).

Scripts/notebook cells to train FCN variants, evaluate Mean IoU and pixel accuracy, and save checkpoints.

Saved experiment artifacts in fcn_experiments/:

Pretrained weights .pth

Training curves curve_*.png

Example segmentation visualizations viz_*.png

summary_table.csv summarizing metrics

Usage (quick):

Open Assignment_2.ipynb.

Ensure dataset path points to your small dataset or adjust download code.

Run training (20 epochs recommended). The notebook logs Mean IoU and pixel accuracy per epoch.

Inspect fcn_experiments/summary_table.csv for comparison between transpose conv and bilinear.

Task 3 — VAE on MNIST (Assignment_3.ipynb & vae_mnist_outputs/)

Goal: Train a convolutional VAE on MNIST using:

Encoder: 3 conv layers → flatten → linear → μ and log(σ²)

latent_dim = 128 (primary), then latent_dim = 256 (retrain)

Decoder: ConvTranspose layers to reconstruct to 28×28

Loss: BCE (or MSE) + KL; Optimizer: Adam; Train 50 epochs for full experiment

What’s included in repo:

Assignment_3.ipynb — full code for loading MNIST, network implementation, training loop, visualizations (reconstructions, sampling, interpolation, PCA of latent means).

vae_mnist_outputs/ — example output images and short-demo checkpoints:

samples_latent128.png

samples_latent256.png

vae128_demo_latent128.pth

vae256_demo_latent256.pth
