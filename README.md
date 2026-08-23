# Quantum Multimodal Framework

Hybrid Quantum-Classical Multimodal Framework for Sentiment Analysis and Image Captioning.

## Overview

Q-MMF leverages Parameterized Quantum Circuits (PQC) for cross-modal fusion in both sentiment analysis and image captioning tasks.

### Key Components

- **Quantum Fusion Layer (QFL)**: 3 variants (Tensor, Attention, Interference) for learning entangled cross-modal representations
- **Quantum Attention Module (QAM)**: Quantum-enhanced cross-attention for image captioning
- **Shared Quantum Feature Space**: Multi-task learning with shared quantum layer

## Installation

```bash
cd quantum-multimodal-framework
pip install -e .
```

## Quick Start

```bash
# Train sentiment model
python -m src.training.train --task sentiment --config src/training/configs/sentiment_config.yaml

# Run all experiments
python experiments/run_all_experiments.py

# Run tests
pytest tests/ -v
```

## Project Structure

```
quantum-multimodal-framework/
├── src/
│   ├── data/          # Dataset loaders
│   ├── encoders/      # BERT text encoder, ResNet image encoder
│   ├── decoders/      # Transformer caption decoder, sentiment head
│   ├── quantum/       # Quantum fusion layers, attention, noise
│   ├── models/        # Full hybrid model
│   ├── training/      # Training loops, loss, optimizer
│   └── evaluation/    # Metrics, visualization
├── experiments/       # Experiment configs and runners
├── tests/             # Unit tests
└── paper/             # LaTeX paper
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- PennyLane >= 0.40
- Transformers >= 4.30
