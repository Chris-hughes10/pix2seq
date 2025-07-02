# Pix2Seq: Object Detection as Sequence Generation

A PyTorch implementation of Pix2Seq for object detection, where detection is formulated as an autoregressive sequence generation task. This implementation supports both standard transformer and Llama-based architectures.

This repo is meant to accompany [this blog post](https://medium.com/@chris.p.hughes10/rethinking-object-detection-as-language-modelling-lessons-from-reimplementing-pix2seq-049104083747?source=friends_link&sk=91f149231a805674299f7a80e556b6f6).

## Features

- **Sequence-based object detection**: Objects represented as token sequences `[y1 x1 y2 x2 class]`
- **Two model architectures**: Standard transformer and Llama-based models with RoPE
- **Comprehensive augmentation**: Image and bounding box augmentation following the original paper
- **Flexible training**: Support for both local development and AzureML cloud training
- **COCO evaluation**: Built-in COCO mAP evaluation and visualization

## Quick Start

### Prerequisites

- Docker (for containerized training)
- VS Code with Dev Containers extension (for development)
- Azure ML CLI v2 (for cloud training)

### Setup

1. **Clone repository**:

2. **Download COCO dataset**:
```bash
cd src/pix2seq
bash get_coco.sh
```

This downloads COCO 2017 images and annotations to `../../data/coco/`.

3. **Open in devcontainer**:
```bash
# Open in VS Code and reopen in container when prompted
code .
```

4. **Development workflow**:
   - **Outside devcontainer**: Run makefile commands for training, building, job submission
   - **Inside devcontainer**: Code editing, debugging, interactive development

### Training

#### Local Development (outside devcontainer)
```bash
# Build the environment for local testing
make build-exp exp=pix2seq

# Run local training (small scale)
make local exp=pix2seq script=train.py script-xargs="--config_file overfit_eval.yaml"

# Run in interactive mode for debugging
make jupyter exp=pix2seq
```

#### AzureML Training (outside devcontainer)
```bash
# Submit training job to AzureML
make job exp=pix2seq
```

### Configuration

Training configurations are in `src/pix2seq/config/`:

- `overfit_eval.yaml`: Small-scale config for debugging/validation
- `train.yaml`: Full training configuration

Key parameters:
```yaml
data:
  num_classes: 80
  max_instances: 100
  image_size: 640

model:
  llama_model: true  # Use Llama architecture
  d_model: 256
  num_encoder_layers: 6
  num_decoder_layers: 6

training:
  num_epochs: 300
  batch_size: 64
  learning_rate: 0.0003
```

## Model Architectures

### Standard Transformer
- Vision Transformer encoder with learned positional embeddings
- Transformer decoder with cross-attention
- Shared embedding projection for output

### Llama-based Model  
- ViT encoder with RoPE positional encoding
- Llama decoder blocks with SwiGLU FFN and RMSNorm
- Rotary position embeddings for better sequence modelling

## Project Structure

```
src/pix2seq/
├── config/                 # Training configurations
├── data/                   # Dataset and tokenization logic
├── model/                  # Model architectures and inference
├── training/               # Training loop and callbacks
├── evaluation/             # COCO evaluation and metrics
├── train.py               # Main training script
└── get_coco.sh           # COCO dataset download script
```

## Key Components

- **TokenProcessor**: Converts bounding boxes to token sequences with coordinate quantization
- **Pix2SeqDataset**: Handles COCO data loading with augmentations
- **BBoxAugmentation**: Generates positive and negative bounding box examples
- **SequenceGenerator**: Autoregressive inference with constraints
- **COCO Evaluation**: mAP calculation with visualization callbacks

## Common Commands

All makefile commands are run **outside the devcontainer**:

```bash
# Create new experiment
make new-exp exp=my_experiment

# Build Docker environment
make build-exp exp=pix2seq

# Local development
make local exp=pix2seq
make jupyter exp=pix2seq
make terminal exp=pix2seq

# Run tests
make test exp=pix2seq

# Submit to AzureML
make job exp=pix2seq

# Format code
make format
```

## Configuration (config.env)

Update `config.env` for your AzureML workspace:

```bash
WORKSPACE=your-workspace
RESOURCE_GROUP=your-resource-group  
CODE_PATH=./src
DOCKER_WORKDIR=/mnt
ISOLATED_RUNS_PATH=./isolated_runs
```

## Monitoring Training

The implementation includes comprehensive logging:

- **Token accuracy metrics**: Position-wise and sequence-level accuracy
- **mAP evaluation**: COCO mean average precision calculation  
- **Prediction visualizations**: Side-by-side ground truth vs predictions
- **AzureML integration**: Automatic metric logging to AzureML

For more detailed help on any command (run **outside devcontainer**):
```bash
make help cmd=<command_name>
```

## Citation

If you use this implementation, please cite the original Pix2Seq paper:

```bibtex
@article{chen2021pix2seq,
  title={Pix2seq: A language modeling framework for object detection},
  author={Chen, Ting and Saxena, Saurabh and Li, Lala and Fleet, David J and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2109.10852},
  year={2021}
}
```
