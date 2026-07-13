"""REINFORCE training script for Pix2Seq.

Fine-tunes a pretrained (MLE) Pix2Seq model with policy gradients on a recall
reward plus IoU-supervised confidences, following "Tuning Computer Vision
Models with Task Rewards" (arXiv:2302.08242).

Supports multi-GPU training via Hugging Face Accelerate: rollouts run on the
unwrapped model without gradients, while the teacher-forced re-scoring forward
pass runs through the DDP-wrapped model so gradients synchronise.

Usage:
    # Single GPU
    python train_rl.py --pretrained_path /path/to/checkpoint.pt

    # Multi-GPU with accelerate
    accelerate launch train_rl.py --pretrained_path /path/to/checkpoint.pt
"""

import datetime
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from func_to_script import load_config_from_yaml, script
from model.model import Pix2SeqModel
from model.modelv2 import LlamaPix2Seq

from data.base_dataset import COCOBaseDataset
from data.dataset import Pix2SeqDataset
from data.tokenizer import LabelCorruptionStrategy, TokenProcessor
from rl import IoUSupervisionLoss, RecallReward, REINFORCETrainer, RLTrainingConfig
from rl.evaluation import evaluate_map
from utils import AzureMLLogger


class RLCollator:
    """Collator for RL training that preserves ground truth boxes as lists."""

    def __init__(self, token_processor: TokenProcessor):
        self.token_processor = token_processor

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch for RL training.

        Returns ground truth boxes and labels as lists (variable length per image)
        rather than padded tensors, which is what the reward function expects.
        """
        images = torch.stack([x["image"] for x in batch])
        image_ids = torch.tensor([x["image_id"] for x in batch])
        orig_sizes = torch.stack([x["orig_image_size"] for x in batch])
        unpadded_sizes = torch.stack([x["unpadded_image_size"] for x in batch])

        # Keep GT boxes and labels as lists (variable length per image)
        gt_boxes_list = []
        gt_labels_list = []

        for item in batch:
            num_boxes = item["num_boxes"].item()
            # Only keep actual boxes (not padding)
            boxes = item["boxes"][:num_boxes]  # [N, 4]
            labels = item["labels"][:num_boxes]  # [N]
            gt_boxes_list.append(boxes)
            gt_labels_list.append(labels)

        return {
            "image": images,
            "image_id": image_ids,
            "orig_image_sizes": orig_sizes,
            "unpadded_image_sizes": unpadded_sizes,
            "gt_boxes": gt_boxes_list,
            "gt_labels": gt_labels_list,
        }


class RLTrainingLogger:
    """Logger for RL training statistics with MLflow/AzureML support.

    Only logs on main process when using distributed training.
    Uses AzureMLLogger for MLflow integration (works locally and on Azure ML).
    """

    def __init__(
        self,
        output_dir: Path,
        log_interval: int = 10,
        accelerator: Accelerator = None,
    ):
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.accelerator = accelerator
        self.log_file = output_dir / "rl_training.log"
        self.stats_file = output_dir / "rl_stats.json"
        self.all_stats = []

        # Initialize MLflow/AzureML logger (only on main process)
        if self._is_main_process():
            self.mlflow_logger = AzureMLLogger()
        else:
            self.mlflow_logger = None

    def _is_main_process(self) -> bool:
        """Check if this is the main process."""
        if self.accelerator is not None:
            return self.accelerator.is_main_process
        return True

    def set_tags(self, tags: Dict[str, str]):
        """Set MLflow tags (e.g., for run configuration)."""
        if self._is_main_process() and self.mlflow_logger is not None:
            self.mlflow_logger.set_tags(tags)

    def log(self, step: int, stats: Dict[str, float]):
        """Log training statistics to console, file, and MLflow."""
        self.all_stats.append({"step": step, **stats})

        if step % self.log_interval == 0 and self._is_main_process():
            # Console and file logging
            msg = (
                f"Step {step}: "
                f"loss={stats['loss/total']:.4f}, "
                f"rl_loss={stats['loss/rl']:.4f}, "
                f"reward={stats['reward/sample_mean']:.4f}, "
                f"baseline={stats['reward/baseline_mean']:.4f}, "
                f"adv={stats['reward/advantage_mean']:.4f}"
            )
            print(msg)

            with open(self.log_file, "a") as f:
                f.write(msg + "\n")

            # MLflow logging - convert stat names to valid metric names
            if self.mlflow_logger is not None:
                mlflow_metrics = {
                    key.replace("/", "_"): value
                    for key, value in stats.items()
                }
                mlflow_metrics["step"] = step
                self.mlflow_logger.log_metrics(mlflow_metrics)

    def log_eval_metrics(self, epoch: int, mAP: float):
        """Log evaluation metrics."""
        if self._is_main_process() and self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics({"eval_mAP": mAP, "epoch": epoch})

    def save_stats(self):
        """Save all statistics to JSON."""
        if self._is_main_process():
            with open(self.stats_file, "w") as f:
                json.dump(self.all_stats, f, indent=2)


def create_datasets(
    config, train_image_dir, train_annotation_file, val_image_dir, val_annotation_file
):
    """Create training and validation datasets."""

    train_ds = COCOBaseDataset(
        train_image_dir,
        train_annotation_file,
        filter_crowd=True,
    )
    eval_ds = COCOBaseDataset(val_image_dir, val_annotation_file, filter_crowd=True)

    # For RL training, we don't need bbox augmentation (no noise boxes)
    train_dataset = Pix2SeqDataset(
        base_dataset=train_ds,
        num_classes=config.data.num_classes,
        training=False,  # Disable bbox augmentation for RL
        max_num_objects=config.data.max_instances,
        image_size=config.data.image_size,
        jitter_scale=config.data.jitter_scale,
        color_jitter_strength=config.data.color_jitter_strength,
    )

    eval_dataset = Pix2SeqDataset(
        base_dataset=eval_ds,
        num_classes=config.data.num_classes,
        training=False,
        max_num_objects=config.data.max_instances,
        image_size=config.data.image_size,
        jitter_scale=config.data.jitter_scale,
        color_jitter_strength=config.data.color_jitter_strength,
    )

    # Load validation annotations for mAP evaluation
    with open(val_annotation_file, "r") as f:
        val_json = json.load(f)

    return train_dataset, eval_dataset, val_json


def create_model(config, token_processor: TokenProcessor, llama_model=False):
    """Create model (without optimizer - we create it separately for RL)."""

    model_instance = Pix2SeqModel if not llama_model else LlamaPix2Seq

    model = model_instance(
        image_size=config.data.image_size,
        patch_size=config.model.patch_size,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        embedding_dim=config.model.d_model,
        num_heads=config.model.nhead,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        drop_path=config.model.drop_path,
        shared_decoder_embedding=config.model.shared_decoder_embedding,
        decoder_output_bias=config.model.decoder_output_bias,
        eos_token_id=token_processor.EOS_TOKEN,
        bos_token_id=token_processor.BOS_TOKEN,
        coord_vocab_shift=token_processor.coord_vocab_shift,
        base_vocab_shift=token_processor.BASE_VOCAB_SHIFT,
        num_quantization_bins=token_processor.quantization_bins,
        max_seq_len=token_processor.max_seq_len,
        token_processor=token_processor,
    )

    return model


def load_pretrained_model(model, checkpoint_path: str, device: torch.device):
    """Load pretrained weights from checkpoint."""
    print(f"Loading pretrained model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print("Pretrained model loaded successfully")

    return model


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_updates: int,
    schedule: str = "constant",
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by a constant or cosine schedule.

    Steps are counted in optimizer updates (i.e. after gradient accumulation).
    """
    if schedule not in ("constant", "cosine"):
        raise ValueError(f"Unknown lr_schedule: {schedule}")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_output_dir(config, output_dir):
    """Create output directory and save config."""
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_dict = {
        "data": dict(config.data) if hasattr(config.data, "__dict__") else config.data,
        "model": dict(config.model) if hasattr(config.model, "__dict__") else config.model,
        "training": dict(config.training) if hasattr(config.training, "__dict__") else config.training,
        "rl": dict(config.rl) if hasattr(config.rl, "__dict__") else config.rl,
    }

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


FILE_PATH = Path(__file__).resolve().parent


@script
def train_rl(
    pretrained_path: str,
    coco_dir: str = "/workspaces/object-detection-rl/data/coco",
    config_file: str = "train_rl.yaml",
    seed: int = 42,
):
    """Main RL training function with multi-GPU support via Accelerate.

    Args:
        pretrained_path: Path to pretrained MLE checkpoint (required)
        coco_dir: Path to COCO dataset directory
        config_file: Path to config file (relative to config/)
        seed: Random seed for reproducibility
    """
    # Load config
    config = load_config_from_yaml((FILE_PATH / "config") / config_file)

    # Initialize Accelerator: gradient accumulation is handled by
    # accelerator.accumulate() in the training loop
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    set_seed(seed)

    # Setup output directory (only on main process)
    output_dir = Path(config.training.output_dir) / datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    if accelerator.is_main_process:
        setup_output_dir(config, output_dir)
    accelerator.wait_for_everyone()

    # Setup paths
    coco_dir = Path(coco_dir)
    train_image_dir = coco_dir / "images/train2017"
    train_annotation_file = coco_dir / "annotations/instances_train2017.json"
    val_image_dir = coco_dir / "images/val2017"
    val_annotation_file = coco_dir / "annotations/instances_val2017.json"

    # Create datasets
    train_dataset, eval_dataset, val_json = create_datasets(
        config,
        train_image_dir,
        train_annotation_file,
        val_image_dir,
        val_annotation_file,
    )

    # Calculate sequence length
    boxes_with_eos_position = train_dataset.max_instances + 1
    tokens_from_boxes = boxes_with_eos_position * 5
    total_seq_len = tokens_from_boxes + 2

    # Create token processor
    token_processor = TokenProcessor(
        quantization_bins=config.tokenization.quantization_bins,
        noise_bbox_weight=config.tokenization.noise_bbox_weight,
        eos_token_weight=config.tokenization.eos_token_weight,
        max_seq_len=total_seq_len,
        num_classes=config.data.num_classes,
        num_special_tokens=10,
        corruption_strategy=LabelCorruptionStrategy.NONE,  # No corruption for RL
    )

    # Create model on CPU first, then load pretrained weights
    model = create_model(config, token_processor, llama_model=config.model.llama_model)
    model = load_pretrained_model(model, pretrained_path, torch.device("cpu"))

    # Create optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.eps,
    )

    # Create data loader
    from torch.utils.data import DataLoader

    collator = RLCollator(token_processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        drop_last=True,
    )

    # LR scheduler stepped per optimizer update (after accumulation)
    updates_per_epoch = math.ceil(
        len(train_loader)
        / (config.training.gradient_accumulation_steps * accelerator.num_processes)
    )
    total_updates = updates_per_epoch * config.training.num_epochs
    lr_scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        total_updates=total_updates,
        schedule=config.training.lr_schedule,
    )

    # Prepare with Accelerator. Prepared optimizer/scheduler automatically skip
    # steps during gradient accumulation
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    # Create RL components
    reward_fn = RecallReward(
        token_processor=token_processor,
        iou_thresholds=config.rl.reward.iou_thresholds,
    )

    iou_loss_fn = None
    if config.rl.iou_supervision.enabled:
        iou_loss_fn = IoUSupervisionLoss(
            min_iou_threshold=config.rl.iou_supervision.min_iou_threshold,
        )

    # Create RL trainer config
    rl_config = RLTrainingConfig(
        algorithm=config.rl.algorithm,
        advantage_type=config.rl.advantage_type,
        num_samples_per_image=config.rl.num_samples_per_image,
        temperature=config.rl.sampling.temperature,
        top_k=config.rl.sampling.top_k,
        top_p=config.rl.sampling.top_p,
        iou_loss_weight=config.rl.iou_supervision.weight if iou_loss_fn else 0.0,
        normalize_advantages=config.rl.normalize_advantages,
        max_grad_norm=config.rl.max_grad_norm,
    )

    # The trainer computes losses; this loop owns backward/clip/step. Pass the
    # wrapped model so the re-scoring forward synchronises gradients under DDP
    trainer = REINFORCETrainer(
        model=model,
        token_processor=token_processor,
        reward_fn=reward_fn,
        iou_loss_fn=iou_loss_fn,
        config=rl_config,
        accelerator=accelerator,
    )

    # Setup logging (only logs on main process)
    logger = RLTrainingLogger(output_dir, log_interval=10, accelerator=accelerator)

    # Set MLflow tags with run configuration
    logger.set_tags({
        "algorithm": config.rl.algorithm,
        "advantage_type": config.rl.advantage_type,
        "num_samples_per_image": str(config.rl.num_samples_per_image),
        "learning_rate": str(config.training.learning_rate),
        "batch_size": str(config.training.batch_size),
        "num_devices": str(accelerator.num_processes),
    })

    # Training info (only print on main process)
    if accelerator.is_main_process:
        effective_batch = (
            config.training.batch_size
            * accelerator.num_processes
            * config.training.gradient_accumulation_steps
        )
        print(f"\n{'='*60}")
        print("Starting REINFORCE Training")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Pretrained model: {pretrained_path}")
        print(f"Learning rate: {config.training.learning_rate}")
        print(f"Batch size per device: {config.training.batch_size}")
        print(f"Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
        print(f"Number of devices: {accelerator.num_processes}")
        print(f"Effective batch size (images per update): {effective_batch}")
        print(f"Samples per image (K): {config.rl.num_samples_per_image}")
        print(f"Advantage type: {config.rl.advantage_type}")
        print(f"Epochs: {config.training.num_epochs}")
        print(f"{'='*60}\n")

    best_mAP = -1
    global_step = 0

    for epoch in range(config.training.num_epochs):
        model.train()
        epoch_stats = []

        for batch in train_loader:
            # Accelerator handles device placement for tensors in batch["image"]
            # But we need to manually move the list items (gt_boxes, gt_labels)
            batch["gt_boxes"] = [b.to(accelerator.device) for b in batch["gt_boxes"]]
            batch["gt_labels"] = [
                labels.to(accelerator.device) for labels in batch["gt_labels"]
            ]

            with accelerator.accumulate(model):
                loss, stats = trainer.compute_losses(batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients and rl_config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(), rl_config.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            stats["lr"] = lr_scheduler.get_last_lr()[0]
            epoch_stats.append(stats)

            # Log
            logger.log(global_step, stats)
            global_step += 1

        # Epoch summary (only on main process)
        if accelerator.is_main_process and epoch_stats:
            avg_stats = {
                key: sum(s[key] for s in epoch_stats) / len(epoch_stats)
                for key in epoch_stats[0].keys()
            }
            print(f"\nEpoch {epoch + 1}/{config.training.num_epochs} Summary:")
            print(f"  Avg Loss: {avg_stats['loss/total']:.4f}")
            print(f"  Avg Reward: {avg_stats['reward/sample_mean']:.4f}")
            print(f"  Avg Baseline: {avg_stats['reward/baseline_mean']:.4f}")

        # Evaluate periodically (only on main process to avoid duplicate work)
        if (epoch + 1) % config.evaluation.eval_frequency == 0:
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                print(f"\nEvaluating at epoch {epoch + 1}...")
                mAP = evaluate_map(
                    model=model,
                    eval_dataset=eval_dataset,
                    collate_fn=collator,
                    token_processor=token_processor,
                    val_json=val_json,
                    device=accelerator.device,
                    max_seq_len=token_processor.max_seq_len,
                    batch_size=config.training.eval_batch_size,
                    confidence_threshold=config.evaluation.confidence_threshold,
                    accelerator=accelerator,
                )
                print(f"  mAP: {mAP:.4f}")

                # Log mAP to MLflow
                logger.log_eval_metrics(epoch + 1, mAP)

                # Save best model
                if mAP > best_mAP:
                    best_mAP = mAP
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": unwrapped_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": lr_scheduler.state_dict(),
                            "mAP": mAP,
                        },
                        output_dir / "best_model.pt",
                    )
                    print(f"  New best model saved! (mAP: {mAP:.4f})")

            accelerator.wait_for_everyone()

    # Save final model (only on main process)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            {
                "epoch": config.training.num_epochs,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
            },
            output_dir / "final_model.pt",
        )

        # Save training stats
        logger.save_stats()

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best mAP: {best_mAP:.4f}")
        print(f"Models saved to: {output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    train_rl()
