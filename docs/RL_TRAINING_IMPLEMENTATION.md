# SCST/REINFORCE Training Implementation for Pix2Seq

## Overview

This document describes the implementation of Self-Critical Sequence Training (SCST) for fine-tuning Pix2Seq models directly on task rewards, based on the paper ["Tuning Computer Vision Models with Task Rewards"](https://arxiv.org/abs/2302.08242).

## Background

### The Problem

Standard Pix2Seq training uses cross-entropy loss (MLE) to predict the next token. However, the model is evaluated using mAP. This creates a mismatch between training objective and evaluation metric.

### The Solution

SCST uses REINFORCE (policy gradient) to fine-tune the model directly on recall-based rewards:

```
Loss = -(reward(sample) - reward(greedy)) × log_prob(sample)
```

Key insight from the paper: **mAP correlates with recall** when the model ranks boxes well, so we use recall as the reward signal.

## Implementation Summary

### Files Modified

| File | Changes |
|------|---------|
| `model/inference.py` | Added `greedy`, `return_log_probs`, `training_mode` parameters |
| `model/model.py` | Updated `infer()` to pass through new parameters |
| `model/modelv2.py` | Same updates as model.py |

### Files Added

| File | Purpose |
|------|---------|
| `rl/__init__.py` | Module exports |
| `rl/rewards.py` | Per-image recall reward computation |
| `rl/iou_loss.py` | Supervised IoU loss for confidence prediction |
| `rl/reinforce.py` | REINFORCE trainer with self-critical baseline |
| `config/train_rl.yaml` | RL training configuration |
| `tests/test_rl.py` | Unit tests |

## Architecture

### SequenceGenerator Extensions

```python
def generate(
    self,
    model,
    images,
    greedy=False,           # NEW: Use argmax instead of sampling
    return_log_probs=False, # NEW: Return token log probabilities
    training_mode=False,    # NEW: Allow gradient flow
):
    ...
```

All parameters have defaults that preserve existing behavior.

### Reward Computation

```python
class RecallReward:
    """Computes per-image recall averaged across IoU thresholds."""

    def __call__(self, sequences, class_logits, gt_boxes, gt_labels):
        # 1. Decode sequences to boxes
        # 2. Match predictions to GT using IoU
        # 3. Compute recall = matched_gt / total_gt
        # 4. Average across thresholds [0.5, 0.55, ..., 0.95]
        return rewards  # [B] tensor
```

### REINFORCE Trainer

```python
class REINFORCETrainer:
    """SCST training loop following simple-ppo structure."""

    def train_step(self, batch):
        # 1. Sample sequences with log probs
        # 2. Generate greedy baseline
        # 3. Compute rewards for both
        # 4. Advantage = sample_reward - greedy_reward
        # 5. Loss = -advantage × log_prob
        # 6. Optional: IoU supervision loss
        return stats
```

### Two-Part Loss (from paper)

1. **REINFORCE Loss**: Optimizes recall
   ```
   L_RL = -(advantage) × Σ log_prob(tokens)
   ```

2. **IoU Supervision Loss**: Trains confidence scores
   ```
   L_IoU = MSE(predicted_confidence, actual_IoU)
   ```

3. **Total Loss**:
   ```
   L = L_RL + λ × L_IoU
   ```

## Usage

### Basic Usage

```python
from rl import REINFORCETrainer, RecallReward, IoUSupervisionLoss, RLTrainingConfig

# Load pretrained MLE model
model = load_checkpoint("path/to/mle_model.pt")

# Create RL components
reward_fn = RecallReward(token_processor)
iou_loss_fn = IoUSupervisionLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Create trainer
trainer = REINFORCETrainer(
    model=model,
    token_processor=token_processor,
    optimizer=optimizer,
    reward_fn=reward_fn,
    iou_loss_fn=iou_loss_fn,
    config=RLTrainingConfig(
        baseline="greedy",
        normalize_advantages=True,
        iou_loss_weight=1.0,
    ),
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        stats = trainer.train_step(batch)
        print(f"Loss: {stats['loss/total']:.4f}")
```

### Batch Format

The trainer expects batches with:
```python
batch = {
    "image": torch.Tensor,           # [B, C, H, W]
    "gt_boxes": List[torch.Tensor],  # List of [M_i, 4] in XYXY format
    "gt_labels": List[torch.Tensor], # List of [M_i] class indices
}
```

## Configuration

See `config/train_rl.yaml` for all options:

```yaml
rl:
  enabled: true
  algorithm: "reinforce"
  baseline: "greedy"

  reward:
    type: "recall"
    iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

  iou_supervision:
    enabled: true
    weight: 1.0

  normalize_advantages: true
  max_grad_norm: 1.0
```

## Training Tips

1. **Start from MLE-pretrained model**: SCST is a fine-tuning technique
2. **Use lower learning rate**: 3e-5 vs 3e-4 for MLE
3. **Smaller batch size**: RL needs memory for sampling twice (sample + greedy)
4. **Monitor advantage**: Should be ~0 mean after normalization
5. **Watch for reward improvement**: sample_reward should increase

## Backwards Compatibility

All changes are backwards compatible:
- New parameters have defaults matching original behavior
- Existing training code works unchanged
- Existing checkpoints load without issues

## Future Extensions

1. **PPO**: Add clipped objective, value function, GAE
2. **Different rewards**: AP@50, F1 score, weighted recall
3. **Curriculum**: Start with high temperature, anneal down

## References

- [Tuning Computer Vision Models with Task Rewards](https://arxiv.org/abs/2302.08242)
- [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563)
- [Pix2seq: A Language Modeling Framework for Object Detection](https://arxiv.org/abs/2109.10852)
