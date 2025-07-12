# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is Twitter's open-sourced machine learning models repository containing:
- **Heavy Ranker**: The "For You" timeline recommendation model using MaskNet architecture
- **TwHIN**: Twitter Heterogeneous Information Network embeddings

**Tech Stack**: Python 3.10, PyTorch 1.13.1, TorchRec 0.3.2 (Meta's recommendation library)

## Development Setup and Commands

### Initial Setup (Linux only)
```bash
# Create and activate virtual environment
./images/init_venv.sh
source $HOME/tml_venv/bin/activate
```

### Heavy Ranker Development
```bash
# Generate random training data
projects/home/recap/script/create_random_data.sh

# Train the model locally
projects/home/recap/script/run_local.sh
```
- Data location: `$HOME/tmp/recap_local_random_data`
- Checkpoints: `$HOME/tmp/runs/recap_local_debug`
- Config: `projects/home/recap/config/local_prod.yaml`

### TwHIN Development
```bash
# Run with Docker
projects/twhin/scripts/docker_run.sh
```

### Testing and Code Quality
```bash
# Run tests with pytest
pytest

# Type checking
mypy .

# Code formatting
black . --line-length 100
```

## Architecture Overview

### Configuration System
- **Pydantic-based** configs with YAML support
- Environment variable substitution (`${HOME}`, `${USER}`)
- Validation via `BaseConfig` with `one_of` and `at_most_one_of` patterns
- Entry point: `core/config/`

### Training Pipeline
- **Pipelined training** with 3 CUDA streams for overlapping compute/transfer
- Distributed model parallelism via TorchRec
- Gradient accumulation and AMP support
- Key class: `TrainPipelineSparseDist` in `core/train_pipeline.py`

### Model Architecture
- **MaskNet**: Parallel attention-based architecture with feature gating
- Multi-task learning with task-specific heads
- Location: `projects/home/recap/model/`

### Embedding System
- Large-scale distributed embeddings via TorchRec
- FP16/FP32 support with vocabulary mapping
- Key class: `LargeEmbeddings` in `common/modules/embedding/`

### Metrics and Logging
- Rank-aware logging (only rank 0 by default)
- Custom metrics: AUROC with Mann-Whitney U-test, position-based metrics
- Stratification support for A/B testing

## Key Development Patterns

1. **Multi-task models**: Use task indices for per-task losses and metrics
2. **Distributed training**: Always consider rank-aware operations
3. **Configuration**: Use Pydantic models for type safety
4. **Testing**: Generate random data for privacy-compliant testing
5. **Performance**: Leverage pipeline parallelism for large models

## Common Tasks

### Adding a New Model
1. Create model class inheriting from `torch.nn.Module`
2. Add configuration in Pydantic format
3. Register in appropriate model factory
4. Implement multi-task support if needed

### Modifying Training Pipeline
1. Check `core/train_pipeline.py` for the main loop
2. Consider stream scheduling for new operations
3. Maintain gradient accumulation compatibility

### Working with Embeddings
1. Use `LargeEmbeddings` wrapper for distributed support
2. Configure sharding strategy in YAML
3. Consider FP16 for memory efficiency

## Important Notes
- **Platform**: Linux with NVIDIA GPU required (macOS not supported)
- **Privacy**: No real data in repo; use random data generators
- **Distributed**: Most operations assume distributed training context