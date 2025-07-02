from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
from evaluation.token_accuracy_evaluator import TokenAccuracyEvaluator
from plotting import show_image_with_boxes
from pytorch_accelerated.callbacks import LogMetricsCallback, TrainerCallback
from pytorch_accelerated.utils import world_process_zero_only
from utils import AzureMLLogger


class AzureMLLoggerCallback(LogMetricsCallback):
    def __init__(self):
        self.logger = AzureMLLogger()

    def on_training_run_start(self, trainer, **kwargs):
        self.logger.set_tags(trainer.run_config.to_dict())

    def log_metrics(self, trainer, metrics):
        if trainer.run_config.is_world_process_zero:
            self.logger.log_metrics(metrics)


class VisualizePredictionsCallback(TrainerCallback):
    """Callback to save visualization of predictions vs ground truth during training."""

    def __init__(
        self,
        output_dir: str,
        category_names: Dict,
        save_freq: int = 5,
        num_images: int = 2,
        images_per_epoch: int = 4,
        confidence_threshold: float = 0.1,
        figsize: tuple = (24, 6),
    ):
        """
        Args:
            output_dir: Directory to save visualizations
            show_image_with_boxes_fn: Function to visualize boxes on images
            category_names: Dictionary mapping class ids to names
            save_freq: Save visualizations every N epochs
            num_images: Number of images to visualize per batch
            confidence_threshold: Minimum confidence score for predictions
            figsize: Figure size for the subplot grid
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_image_with_boxes = show_image_with_boxes
        self.category_names = category_names
        self.save_freq = save_freq
        self.num_images = num_images
        self.confidence_threshold = confidence_threshold
        self.images_per_epoch = images_per_epoch
        self.figsize = figsize
        self._current_epoch_images = 0

    def on_eval_epoch_start(self, trainer, **kwargs):
        self._current_epoch_images = 0

    def visualize_predictions(self, images, predictions, targets, epoch: int):
        """Create side-by-side visualization of predictions vs ground truth."""
        batch_size = len(images)
        num_images = min(self.num_images, batch_size)

        # Create figure with 2 columns (GT and Pred) per image
        fig, axes = plt.subplots(num_images, 2, figsize=self.figsize)
        if num_images == 1:
            axes = axes.reshape(1, 2)

        for idx in range(num_images):
            image = images[idx]

            # Ground truth visualization
            self.show_image_with_boxes(
                image=image,
                boxes=targets["boxes"][idx],
                labels=targets["labels"][idx],
                title="Ground Truth",
                category_names=self.category_names,
                ax=axes[idx, 0],
                normalized_boxes=True,
                box_color="green",
                label_prefix="GT",
            )

            # Filter predictions by confidence
            scores = predictions["scores"][idx]
            mask = scores > self.confidence_threshold
            pred_boxes = predictions["boxes"][idx][mask]
            pred_labels = predictions["labels"][idx][mask]

            # Prediction visualization
            self.show_image_with_boxes(
                image=image,
                boxes=pred_boxes,
                labels=pred_labels,
                title=f"Predictions (conf > {self.confidence_threshold})",
                category_names=self.category_names,
                ax=axes[idx, 1],
                normalized_boxes=True,
                box_color="red",
                label_prefix="Pred",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir
            / f"predictions_epoch_{epoch}_batch_{self._current_epoch_images}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

    @world_process_zero_only
    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        """Save visualizations at specified frequency."""
        current_epoch = trainer.run_history.current_epoch

        if (
            current_epoch % self.save_freq == 0
            and self._current_epoch_images <= self.images_per_epoch
        ):
            # Get images and ground truth
            images = batch["image"]
            boxes = batch["boxes"]
            labels = batch["labels"]

            if batch_output["pred_seq"] is None:
                return

            # Process predictions from batch_output
            pred_seq = batch_output["pred_seq"]
            class_logits = batch_output["class_logits"]

            # Decode predictions
            boxes_list, labels_list, scores_list = (
                trainer.token_processor.decode_tokens(
                    tokens=pred_seq, token_scores=class_logits
                )
            )

            predictions = {
                "boxes": boxes_list,
                "labels": labels_list,
                "scores": scores_list,
                "image_ids": batch["image_id"].tolist(),
            }

            targets = {"boxes": boxes, "labels": labels}

            self.visualize_predictions(images, predictions, targets, current_epoch)
            self._current_epoch_images += 1


class TokenAccuracyCallback(TrainerCallback):
    """Callback wrapper for token accuracy evaluation."""

    def __init__(
        self, pad_token_id: int = 0, eos_token_id: int = 2, print_samples: bool = False
    ):
        self.evaluator = TokenAccuracyEvaluator(
            pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )
        self.print_samples = print_samples
        if self.print_samples:
            self.samples = []

    def _update_run_history(self, trainer, metrics: Dict[str, float], prefix: str):
        """Update trainer's run history with current metrics."""
        # Update batch metrics
        for name, value in metrics.items():
            trainer.run_history.update_metric(f"{prefix}_batch_{name}_accuracy", value)

        # Update epoch metrics if available
        epoch_metrics = self.evaluator.get_epoch_metrics()
        for name, value in epoch_metrics.items():
            trainer.run_history.update_metric(f"{prefix}_epoch_{name}_accuracy", value)

    def on_train_epoch_start(self, trainer, **kwargs):
        self.evaluator.reset_metrics()

    def on_train_step_end(self, trainer, batch: Dict, batch_output: Dict, **kwargs):
        metrics = self.evaluator.compute_batch_metrics(
            batch_output["logits"],
            batch_output["target_seq"],
            batch_output["token_weights"],
            gather_fn=trainer.gather,
        )
        self._update_run_history(trainer, metrics, "train")

    def on_eval_epoch_start(self, trainer, **kwargs):
        self.evaluator.reset_metrics()
        if self.print_samples:
            self.samples = []

    def on_eval_step_end(self, trainer, batch: Dict, batch_output: Dict, **kwargs):
        if self.print_samples and len(self.samples) < 2:
            # Get unflattened predictions from logits and corresponding target sequence
            pred_seq = torch.argmax(batch_output["logits"], dim=-1)
            targets = batch["target_seq"]
            num_to_add = min(2 - len(self.samples), pred_seq.size(0))
            for i in range(num_to_add):
                self.samples.append(
                    (pred_seq[i].detach().cpu(), targets[i].detach().cpu())
                )

        metrics = self.evaluator.compute_batch_metrics(
            batch_output["logits"].view(-1, batch_output["logits"].size(-1)),
            batch["target_seq"].view(-1),
            batch["token_weights"].view(-1),
            gather_fn=trainer.gather,
        )
        self._update_run_history(trainer, metrics, "eval")

    def on_train_epoch_end(self, trainer, **kwargs):
        """Log final epoch metrics."""
        epoch_metrics = self.evaluator.get_epoch_metrics()
        for name, value in epoch_metrics.items():
            trainer.run_history.update_metric(f"train_epoch_{name}_accuracy", value)

    def on_eval_epoch_end(self, trainer, **kwargs):
        """Log final epoch metrics."""
        epoch_metrics = self.evaluator.get_epoch_metrics()
        for name, value in epoch_metrics.items():
            trainer.run_history.update_metric(f"eval_epoch_{name}_accuracy", value)
        if self.print_samples:
            try:
                trainer.print("############################################")
                for idx, (pred, target) in enumerate(self.samples):
                    trainer.print(
                        f"Eval sample {idx} -> Predicted: {pred[:11].tolist()}, Target: {target[:11].tolist()}"
                    )
                trainer.print("############################################")
            except Exception as e:
                pass


class LearningRateTrackerCallback(TrainerCallback):
    """
    A callback that tracks the learning rate from the optimizer and adds it to the trainer's run history at the end of each training epoch.
    """

    def on_train_epoch_end(self, trainer, **kwargs):
        # For simplicity, we assume the learning rate is in the first parameter group.
        lr = trainer.optimizer.param_groups[0]["lr"]
        # Update run history with the learning rate for this epoch.
        trainer.run_history.update_metric("learning_rate", lr)
