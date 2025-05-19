# PixelCraft-Attention-based-GAN-with-Residual-Detail-Fusion-Learning-and-YOLO-Detection
**COMP4423 Project: PixelCraft: Attention based GAN with Residual Detail Fusion Learning and YOLO Detection**
# Attention-based GAN with Residual Detail Fusion Learning (AttnDetailNet)

## Overview
The proposed model, AttnDetailNet, addresses the issue of generating images with insufficient details such as cars or people, while maintaining the overall style of the target data. Inspired by residual learning approaches, the model incorporates a **DetailNet** to refine the output of the **MainNet**, using residuals between the generated images and target images to improve accuracy.

## Key Components

### 1. MainNet (UNetGeneratorPlus)
A U-Net-based architecture designed for global structure and content generation:
- **Deeper Encoding-Decoding Paths**: Captures features at multiple scales.
- **Residual Blocks at Bottleneck**: Mitigates vanishing gradient issues.
- **Self-Attention Modules**: Improves long-range dependency modeling.
- **Skip Connections**: Preserves fine spatial details.

### 2. DetailNet
A lightweight complementary network for high-frequency detail refinement:
- **Shallow Encoder**: Limited downsampling (factor of 4) to retain spatial information critical for fine details.
- **Residual and Attention Processing**: Enhances feature representation while ensuring coherence in detail generation.
- **Detail-Preserving Decoder**: Bilinear upsampling for smooth interpolation and artifact reduction.

### 3. Generator Fusion
Combines MainNet and DetailNet outputs through a learnable weight parameter (α) to balance global structure and fine detail contributions.

### 4. Discriminator (PatchDiscriminator)
A fully-convolutional discriminator that evaluates image patches instead of entire images:
- **Markovian Design**: Focuses on local textures and details.
- **Instance Normalization**: Handles style variations.
- **LeakyReLU Activations**: Prevents gradient vanishing during training.

## Building Blocks

### ConvINReLU
A foundational module with convolution, instance normalization, and ReLU activation:
- **Design Rationale**: Instance normalization preserves style information better than batch normalization, critical for image generation tasks.

### Residual Block
Implements residual learning (`ResBlock(x) = x + F(x)`):
- **Design Rationale**: Residual connections improve gradient flow and stabilize training in deep networks.

### Self-Attention Module
Implements self-attention for global feature coordination:
- **Design Rationale**: Enhances spatial consistency and texture coherence by modeling long-range dependencies.

### Upsampling Block
Handles feature map upsampling:
- **Design Rationale**: Nearest-neighbor upsampling followed by convolution reduces artifacts.

## Architecture Details

### MainNet
- **Encoder**: Progressive downsampling with 8 layers, doubling feature channels while halving spatial dimensions.
- **Bottleneck**: Three sequential residual blocks at the lowest resolution for abstract feature refinement.
- **Decoder**: Progressive upsampling with skip connections for spatial detail preservation.

### DetailNet
- **Encoder**: Shallow structure with reduced channel count and limited downsampling.
- **Bottleneck**: Residual blocks and self-attention for detail enhancement.
- **Decoder**: Symmetric structure with bilinear upsampling for smooth detail restoration.

### Fusion Mechanism
Combines MainNet and DetailNet outputs adaptively using a learnable parameter α to produce globally coherent and detailed images.

### Discriminator
- **Patch-Based Design**: Classifies image patches for efficient texture and detail evaluation.
- **Progressive Strides**: Reduces spatial dimensions while extracting deep features.

## Advantages
- **Global and Local Optimization**: MainNet focuses on overall structure, while DetailNet refines high-frequency details.
- **Adaptive Fusion**: Learnable weighting ensures optimal integration of structural and detail information.
- **Efficient Discriminator**: Patch-based approach provides precise feedback for texture and detail generation.

The proposed Attention-based GAN with Residual Detail Fusion Learning achieves high-quality image generation with enriched textures and consistent global structures.
