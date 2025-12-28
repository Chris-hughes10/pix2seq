# Plan: SCST/REINFORCE Training for Pix2Seq

Based on "Tuning Computer Vision Models with Task Rewards" (arXiv:2302.08242)

## Overview

Implement Self-Critical Sequence Training (SCST) to fine-tune the pix2seq model directly on recall-based rewards, bridging the gap between cross-entropy training and mAP evaluation.

---

## Phase 1: Modify SequenceGenerator

**File:** `src/pix2seq/model/inference.py`

### Changes Required:

1. **Add `greedy` parameter to `generate()`**
   - When `greedy=True`: use `argmax` instead of `multinomial`
   - When `greedy=False`: use current stochastic sampling

2. **Add `return_log_probs` parameter to `generate()`**
   - Track log probabilities during generation
   - Return `log_probs: [B, S]` tensor alongside sequences

3. **Remove `inference_mode()` when training**
   - Add `training_mode` parameter to allow gradient flow
   - Only disable gradients for greedy baseline computation

4. **Modify `_sample_next_tokens()`**
   - Return both sampled tokens AND their log probabilities
   - Use `log_softmax` to compute log probs efficiently

### Interface After Changes:

```python
def generate(
    self,
    model: torch.nn.Module,
    images: torch.Tensor,
    greedy: bool = False,
    return_log_probs: bool = False,
    training_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns:
        sequences: [B, S] generated token sequences
        class_logits: [B, N, V] logits for class tokens
        features: Optional encoded features
        log_probs: Optional [B, S] log probability per token (if return_log_probs=True)
    """
```

---

## Phase 2: Reward Computation

**File:** `src/pix2seq/rl/rewards.py` (NEW)

### Components:

1. **`compute_per_image_recall()`**
   - Match predicted boxes to ground truth using IoU
   - Compute recall = matched_gt / total_gt
   - Average across multiple IoU thresholds [0.5, 0.55, ..., 0.95]

2. **`compute_batch_rewards()`**
   - Decode token sequences to boxes
   - Compute recall for each image in batch
   - Apply optional class frequency weighting

3. **Box matching logic**
   - Greedy matching: each GT matched to highest-IoU prediction (above threshold)
   - Handle edge cases: no predictions, no GT boxes

### Interface:

```python
class RecallReward:
    def __init__(
        self,
        token_processor: TokenProcessor,
        iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        class_weights: Optional[torch.Tensor] = None,
    ):
        ...

    def __call__(
        self,
        sequences: torch.Tensor,      # [B, S] token sequences
        class_logits: torch.Tensor,   # [B, N, V] for confidence scores
        gt_boxes: List[torch.Tensor], # List of [N_i, 4] ground truth boxes
        gt_labels: List[torch.Tensor] # List of [N_i] ground truth labels
    ) -> torch.Tensor:                # [B] per-image rewards
        ...
```

---

## Phase 3: IoU Supervision Loss

**File:** `src/pix2seq/rl/iou_loss.py` (NEW)

### Purpose:
Train the model's confidence scores to predict actual IoU with ground truth.
This helps with box ranking/NMS at test time.

### Components:

1. **`compute_best_iou_per_prediction()`**
   - For each predicted box, find best matching GT box
   - Return IoU value (0 if no match above threshold)

2. **`IoUSupervisionLoss`**
   - MSE loss between predicted confidence and actual IoU
   - Only apply to non-padding predictions

### Interface:

```python
class IoUSupervisionLoss:
    def __init__(self, min_iou_threshold: float = 0.0):
        ...

    def __call__(
        self,
        pred_boxes: List[torch.Tensor],       # [N_i, 4] per image
        pred_confidences: List[torch.Tensor], # [N_i] per image
        gt_boxes: List[torch.Tensor],         # [M_i, 4] per image
    ) -> torch.Tensor:                        # Scalar loss
        ...
```

---

## Phase 4: REINFORCE Trainer

**File:** `src/pix2seq/rl/reinforce.py` (NEW)

### Structure (following simple-ppo pattern):

```python
class REINFORCETrainer:
    """REINFORCE with self-critical baseline (SCST)."""

    def __init__(
        self,
        model: nn.Module,
        token_processor: TokenProcessor,
        optimizer: torch.optim.Optimizer,
        reward_fn: RecallReward,
        iou_loss_fn: Optional[IoUSupervisionLoss] = None,
        iou_loss_weight: float = 1.0,
        baseline: str = "greedy",  # "greedy" or "sample_mean"
        entropy_coeff: float = 0.0,  # Optional entropy bonus
        max_grad_norm: float = 1.0,
    ):
        ...

    def collect_samples(
        self,
        images: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Collect sampled and greedy sequences with rewards."""
        # 1. Sample sequences with log probs
        sampled_seqs, sampled_logits, _, sampled_log_probs = self.generate(
            images, greedy=False, return_log_probs=True, training_mode=True
        )

        # 2. Greedy baseline (no gradients)
        with torch.no_grad():
            greedy_seqs, greedy_logits, _, _ = self.generate(
                images, greedy=True, return_log_probs=False
            )

        # 3. Compute rewards
        sample_rewards = self.reward_fn(sampled_seqs, sampled_logits, gt_boxes, gt_labels)
        greedy_rewards = self.reward_fn(greedy_seqs, greedy_logits, gt_boxes, gt_labels)

        return {
            "sampled_seqs": sampled_seqs,
            "sampled_log_probs": sampled_log_probs,
            "sampled_logits": sampled_logits,
            "sample_rewards": sample_rewards,
            "greedy_rewards": greedy_rewards,
        }

    def compute_advantages(
        self,
        sample_rewards: torch.Tensor,
        greedy_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantage using self-critical baseline."""
        return sample_rewards - greedy_rewards  # [B]

    def compute_reinforce_loss(
        self,
        log_probs: torch.Tensor,    # [B, S]
        advantages: torch.Tensor,   # [B]
        mask: torch.Tensor,         # [B, S] valid token mask
    ) -> torch.Tensor:
        """REINFORCE policy gradient loss."""
        # Sum log probs over sequence (only valid tokens)
        sequence_log_probs = (log_probs * mask).sum(dim=1)  # [B]

        # Normalize advantages (optional, reduces variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # REINFORCE: maximize expected reward
        loss = -(advantages.detach() * sequence_log_probs).mean()

        return loss

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        images = batch["image"]
        gt_boxes = batch["gt_boxes"]
        gt_labels = batch["gt_labels"]

        # Collect samples
        samples = self.collect_samples(images, gt_boxes, gt_labels)

        # Compute advantages
        advantages = self.compute_advantages(
            samples["sample_rewards"],
            samples["greedy_rewards"]
        )

        # Compute REINFORCE loss
        mask = self.get_valid_token_mask(samples["sampled_seqs"])
        rl_loss = self.compute_reinforce_loss(
            samples["sampled_log_probs"],
            advantages,
            mask
        )

        # Optional: IoU supervision loss
        total_loss = rl_loss
        iou_loss = torch.tensor(0.0)
        if self.iou_loss_fn is not None:
            pred_boxes, pred_confs = self.decode_predictions(
                samples["sampled_seqs"],
                samples["sampled_logits"]
            )
            iou_loss = self.iou_loss_fn(pred_boxes, pred_confs, gt_boxes)
            total_loss = rl_loss + self.iou_loss_weight * iou_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "rl_loss": rl_loss.item(),
            "iou_loss": iou_loss.item(),
            "advantage_mean": advantages.mean().item(),
            "sample_reward_mean": samples["sample_rewards"].mean().item(),
            "greedy_reward_mean": samples["greedy_rewards"].mean().item(),
        }
```

---

## Phase 5: Configuration

**File:** `src/pix2seq/config/train_rl.yaml` (NEW)

```yaml
# Base configuration
defaults:
  - train  # Inherit from standard training config

# RL-specific settings
rl:
  enabled: true
  algorithm: "reinforce"  # "reinforce" or "ppo" (future)

  # Reward settings
  reward:
    type: "recall"
    iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    use_class_weights: false

  # REINFORCE settings
  reinforce:
    baseline: "greedy"  # "greedy" or "sample_mean"
    normalize_advantages: true
    entropy_coeff: 0.0

  # IoU supervision
  iou_supervision:
    enabled: true
    weight: 1.0

  # Training settings
  pretrained_path: "path/to/mle_pretrained_model.pt"  # Start from MLE-trained model
  learning_rate: 0.00003  # Lower LR for fine-tuning
  max_grad_norm: 1.0
  num_epochs: 50
```

---

## Phase 6: Integration

**File:** `src/pix2seq/train_rl.py` (NEW)

Main entry point for RL training:

```python
def main():
    # 1. Load config
    # 2. Load pretrained MLE model
    # 3. Create reward function and IoU loss
    # 4. Create REINFORCE trainer
    # 5. Training loop with evaluation callbacks
    # 6. Save best model based on mAP
```

---

## Implementation Order

1. **Phase 1: SequenceGenerator modifications** (~30 lines)
   - Add greedy flag
   - Add log prob tracking
   - Add training_mode flag

2. **Phase 2: Reward computation** (~100 lines)
   - Per-image recall
   - Box matching
   - Multi-threshold averaging

3. **Phase 3: IoU supervision loss** (~50 lines)
   - Best IoU computation
   - MSE loss

4. **Phase 4: REINFORCE trainer** (~200 lines)
   - Following simple-ppo structure
   - Collect samples, compute advantages, update policy

5. **Phase 5: Config and integration** (~100 lines)
   - New config file
   - Training script

6. **Phase 6: Testing** (~50 lines)
   - Unit tests for reward computation
   - Integration test for training loop

---

## Future: PPO Extension

After REINFORCE is working:

1. Add value head to model (or separate critic network)
2. Implement GAE for advantage estimation
3. Add clipped surrogate objective
4. Add value function loss
5. Multiple epochs over collected samples

---

## Success Criteria

1. Training loop runs without errors
2. Rewards increase over training
3. mAP improves compared to MLE-only baseline
4. Training is stable (no NaN losses, gradients don't explode)

---

## Estimated Effort

| Phase | Lines of Code | Complexity |
|-------|---------------|------------|
| Phase 1 | ~30 | Low |
| Phase 2 | ~100 | Medium |
| Phase 3 | ~50 | Low |
| Phase 4 | ~200 | Medium |
| Phase 5 | ~100 | Low |
| Phase 6 | ~50 | Low |
| **Total** | **~530** | |
