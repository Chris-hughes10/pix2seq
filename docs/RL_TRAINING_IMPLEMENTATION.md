# REINFORCE Training for Pix2Seq

## Overview

This document describes the RL fine-tuning implementation for Pix2Seq, based on
["Tuning Computer Vision Models with Task Rewards"](https://arxiv.org/abs/2302.08242)
(Pinto et al., ICML 2023).

## Background

### The Problem

Standard Pix2Seq training uses cross-entropy loss (MLE) to predict the next
token, but the model is evaluated with mAP. This creates a mismatch between the
training objective and the evaluation metric.

### The Solution (two-part, from the paper)

1. **REINFORCE on a recall reward**: fine-tune the sequence policy directly on
   per-image recall (averaged over the COCO IoU thresholds).
2. **IoU-supervised confidences**: train the class-token confidence to predict
   the IoU of its box with ground truth, so box ranking (and therefore mAP)
   improves alongside recall.

```
L = -(advantage) * sum log p(sampled tokens)  +  lambda * MSE(confidence, IoU)
```

The paper reports mAP improving from 39.2 to 54.3 with this recipe.

## Architecture: rollout + teacher-forced re-scoring

Backpropagating through autoregressive generation is not possible here (the KV
cache writes its buffers in place, which breaks autograd) and would be
memory-prohibitive. Instead, each training step does:

1. **Rollout (no gradients)**: the unwrapped model samples K sequences per
   image under `torch.inference_mode()` with the KV cache. Images are encoded
   once; encoded features are repeated K times (`num_samples` in
   `generate()`/`infer()`). Sampling uses the **full temperature distribution**
   (top-k/top-p disabled — truncated sampling would bias the policy gradient).
2. **Rewards + advantages**: sequences decode to boxes; per-image recall is
   averaged over IoU thresholds; advantages come from a group baseline (below).
3. **Re-scoring (gradients)**: one teacher-forced forward pass — the same code
   path as MLE training — through the *accelerator-wrapped* model recomputes
   log-probs of the sampled tokens. The generation-time constraint masks are
   rebuilt vectorized (`rl/rescoring.py:build_constraint_masks`) and the same
   temperature applied, so the re-scored distribution equals the sampling
   distribution exactly (tested to 1e-4 in `tests/test_rescoring.py`).
4. **Losses**: REINFORCE loss over tokens up to the first EOS, plus the
   IoU-supervision MSE on differentiable class-token confidences from the same
   forward pass.

Because the gradient-carrying forward pass goes through the wrapped model's
`__call__`, DDP gradient synchronisation works under `accelerate launch`.

## Advantage baselines (`rl.advantage_type`)

| Type | Baseline | Notes |
|------|----------|-------|
| `loo` (default) | mean reward of the *other* K-1 samples | Paper-style multi-sample baseline; requires K >= 2 |
| `mean` | group mean | requires K >= 2 |
| `grpo` | (r - group mean) / (group std + eps) | GRPO advantages; requires K >= 2 |
| `greedy` | reward of a greedy decode (SCST) | works with K = 1, costs an extra greedy pass |

### GRPO status

`advantage_type: grpo` already gives GRPO's advantage structure. The remaining
GRPO/PPO ingredients (per-token ratio clipping against the rollout's
generation-time log-probs, optional KL to a reference model) slot into
`compute_policy_gradient_loss` via its `old_per_token_log_probs`/`clip_range`
parameters — the rollout already returns generation-time log-probs for this.
Setting `clip_range` currently raises `NotImplementedError`.

## Files

| File | Purpose |
|------|---------|
| `rl/rescoring.py` | Teacher-forced re-scoring: constraint-mask rebuild, valid-token mask, differentiable log-probs and class confidences |
| `rl/rewards.py` | Per-image recall reward (class-aware, IoU-threshold averaged) |
| `rl/iou_loss.py` | MSE between class-token confidence and best-matching-GT IoU |
| `rl/reinforce.py` | Config, advantages, policy-gradient loss, `REINFORCETrainer` |
| `rl/evaluation.py` | Greedy-decoding mAP evaluation reusing the base trainer's box scaling |
| `train_rl.py` | Training loop: gradient accumulation, LR warmup, checkpointing, MLflow logging |
| `config/train_rl.yaml` | RL training configuration |
| `tests/` | Synthetic test suite (see `test_rescoring.py` for the core invariant) |

`model/inference.py` gained `greedy`, `return_log_probs` and `num_samples`
parameters on `generate()` (all defaults preserve existing behaviour), and
`model.forward()` gained `num_tgt_per_image` for scoring K sequences per image
with one encoder pass.

Note: fixing the RL sampling also fixed a pre-existing generation bug — the
dynamic ymax/xmax constraint masks were previously unioned across the batch, so
per-sample `ymax > ymin`/`xmax > xmin` was not enforced for batch sizes > 1.
Constraints are now correctly per-sample, which can slightly change eval
generation for existing checkpoints.

## Usage

```bash
# Single GPU
python train_rl.py --pretrained_path /path/to/mle_checkpoint.pt

# Multi-GPU with Accelerate
accelerate launch train_rl.py --pretrained_path /path/to/mle_checkpoint.pt
```

Batch format expected by `REINFORCETrainer.compute_losses` (produced by
`RLCollator`):

```python
batch = {
    "image": torch.Tensor,           # [B, C, H, W]
    "gt_boxes": List[torch.Tensor],  # List of [M_i, 4], normalized XYXY
    "gt_labels": List[torch.Tensor], # List of [M_i] class indices
}
```

## Configuration

See `config/train_rl.yaml`. Key settings:

```yaml
training:
  batch_size: 8                    # images per device; K sequences each
  gradient_accumulation_steps: 4
  warmup_steps: 500
  lr_schedule: "constant"          # or "cosine"

rl:
  advantage_type: "loo"            # loo | mean | grpo | greedy
  num_samples_per_image: 4         # K
  sampling:
    temperature: 1.0
    top_k: 0                       # must stay 0
    top_p: 0.0                     # must stay 0
  iou_supervision:
    enabled: true
    weight: 1.0
```

## Training tips

1. **Start from an MLE-pretrained model** — RL is a fine-tuning step.
2. **Keep dropout at 0** — rollouts sample in eval mode while re-scoring runs
   in train mode; with dropout the two distributions diverge (the trainer
   warns about this).
3. **Watch `reward/sample_mean`** — it should climb; `policy/tokens_per_seq_mean`
   shows whether the policy is collapsing to early EOS.
4. **Memory**: the re-scoring pass materialises `[B*K, S-1, V]` logits; lower
   `batch_size` or K if needed.

## Testing

```bash
python -m pytest tests/ -q
```

The suite runs on CPU with a tiny randomly initialised model. The core tests:

- `test_rescoring.py`: re-scored log-probs equal generation log-probs.
- `test_constraint_masks.py`: mask rebuild matches generation per step.
- `test_rl_integration.py::TestOverfitToyBatch`: reward increases when
  overfitting a toy batch end-to-end.

## References

- [Tuning Computer Vision Models with Task Rewards](https://arxiv.org/abs/2302.08242)
- [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563)
- [Pix2seq: A Language Modeling Framework for Object Detection](https://arxiv.org/abs/2109.10852)
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
