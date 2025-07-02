import warnings
from typing import Any, Dict

import torch
from pytorch_accelerated.trainer import Trainer

PADDING_VALUE = -1


def scale_bboxes_to_original_image_size(
    xyxy_boxes, resized_hw, original_hw, is_padded=True, normalized=True
):
    scaled_boxes = xyxy_boxes.clone()

    # First denormalize if boxes are in [0,1] range
    if normalized:
        scaled_boxes[:, [0, 2]] *= resized_hw[1]  # x coords * width
        scaled_boxes[:, [1, 3]] *= resized_hw[0]  # y coords * height

    scale_ratio = resized_hw[0] / original_hw[0], resized_hw[1] / original_hw[1]

    if is_padded:
        # remove padding
        pad_scale = min(scale_ratio)
        padding = (
            (resized_hw[1] - original_hw[1] * pad_scale) / 2,
            (resized_hw[0] - original_hw[0] * pad_scale) / 2,
        )
        scaled_boxes[:, [0, 2]] -= padding[0]  # x padding
        scaled_boxes[:, [1, 3]] -= padding[1]  # y padding
        scale_ratio = (pad_scale, pad_scale)

    scaled_boxes[:, [0, 2]] /= scale_ratio[1]
    scaled_boxes[:, [1, 3]] /= scale_ratio[0]

    # Clip bounding xyxy bounding boxes to image shape (height, width)
    scaled_boxes[:, 0].clamp_(0, original_hw[1])  # x1
    scaled_boxes[:, 1].clamp_(0, original_hw[0])  # y1
    scaled_boxes[:, 2].clamp_(0, original_hw[1])  # x2
    scaled_boxes[:, 3].clamp_(0, original_hw[0])  # y2

    return scaled_boxes


class Pix2SeqTrainer(Trainer):
    """PyTorch trainer for Pix2Seq with optimal integration with parent class."""

    def __init__(
        self,
        model,
        optimizer,
        top_p: float = 0.4,
        top_k: int = 0,
        temperature: float = 1.0,
        run_eval_freq: int = 10,
        token_processor: Dict[str, Any] = None,
        output_dir: str = None,
        **kwargs,
    ):
        # Initialize loss function here instead of taking as argument
        loss_func = torch.nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=0,  # Ignore padding index (tokenized padding)
            # label_smoothing=0.1,
        )

        super().__init__(model, loss_func, optimizer, **kwargs)

        self.token_processor = token_processor
        self.max_seq_len = self.token_processor.max_seq_len
        self.run_eval_freq = run_eval_freq

        # Generation parameters
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

    def train_epoch_start(self):
        self.current_step = 0
        self.collate_fn.set_mode(is_training=True)
        return super().train_epoch_start()

    def eval_epoch_start(self):
        self.current_batch = 0
        self.collate_fn.set_mode(is_training=False)
        return super().eval_epoch_start()

    def calculate_train_batch_loss(self, batch) -> dict:
        """Calculate training loss exactly matching TF implementation."""

        images = batch["image"]  # [B,3,H,W]
        input_seq = batch["input_seq"]  # [B,S]
        target_seq = batch["target_seq"]  # [B,S]
        token_weights = batch["token_weights"]  # [B,S]
        input_padding_mask = batch["input_padding_mask"]
        # Forward pass returns logits [B,S,V]
        logits = self.model(images, input_seq, tgt_padding_mask=input_padding_mask)

        # Reshape for loss calculation
        B, S, V = logits.shape
        logits = logits.view(-1, V)  # [B*S,V]
        target_seq = target_seq.view(-1)  # [B*S]
        token_weights = token_weights.view(-1)  # [B*S]

        loss = self._calculate_loss(logits, target_seq, token_weights)

        return {
            "loss": loss,
            "logits": logits,
            "target_seq": target_seq,
            "token_weights": token_weights,
            "model_outputs": logits.view(B, S, V),
            "batch_size": B,
        }

    def _calculate_loss(self, logits, target_seq, token_weights) -> dict:
        # Calculate per-token cross entropy
        loss = self.loss_func(logits, target_seq)  # [B*S]

        # Apply token weights
        weighted_loss = loss * token_weights  # [B*S]

        # Count non-padding tokens for normalization
        non_padding = (target_seq != self.token_processor.PADDING_TOKEN).float()
        num_valid = non_padding.sum().clamp(min=1e-8)

        # Normalize by number of valid tokens
        loss = weighted_loss.sum() / num_valid

        return loss

    def calculate_eval_batch_loss(self, batch) -> dict:
        """Calculate evaluation loss and process predictions with class scores."""
        with torch.no_grad():
            images = batch["image"]
            input_seq = batch["input_seq"]
            target_seq = batch["target_seq"]
            token_weights = batch["token_weights"]
            input_padding_mask = batch["input_padding_mask"]
            image_ids = batch["image_id"]
            original_image_sizes = batch["orig_image_sizes"]

            # Forward pass for loss calculation
            logits = self.model(images, input_seq, tgt_padding_mask=input_padding_mask)
            # Reshape for consistent loss calculation
            B, S, V = logits.shape
            logits = logits.view(-1, V)  # [B*S,V]
            target_seq = target_seq.view(-1)  # [B*S]
            token_weights = token_weights.view(-1)  # [B*S]

            loss = self._calculate_loss(logits, target_seq, token_weights)

            predictions_list = []
            pred_seq = None
            class_logits = None

            if self.run_history.current_epoch % self.run_eval_freq == 0:
                try:
                    unwrapped_model = self.get_model()

                    pred_seq, class_logits, _ = unwrapped_model.infer(
                        images=images,
                        max_seq_len=self.max_seq_len,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                    )

                    # Post-process predictions
                    boxes_list, labels_list, scores_list = (
                        self.token_processor.post_process_sequences(
                            sequences=pred_seq,
                            class_logits=class_logits,
                            confidence_threshold=0.05,  # Adjust as needed
                        )
                    )

                    resized_image_sizes = torch.as_tensor(
                        images.shape[2:], device=original_image_sizes.device
                    )[None].repeat(len(scores_list), 1)

                    # Format predictions as tensor for evaluation callback
                    # Expected format: [xmin, ymin, xmax, ymax, score, class_id, image_id]
                    self._format_predictions(
                        boxes_list,
                        labels_list,
                        scores_list,
                        image_ids,
                        original_image_sizes,
                        resized_image_sizes,
                        predictions_list,
                    )
                    self.current_batch += 1
                except Exception as e:
                    print(f"Error generating predictions: {e}")
                    print(f"Current epoch: {self.run_history.current_epoch}")

            # Stack all predictions if we have any
            if predictions_list:
                all_predictions = torch.cat(predictions_list, dim=0)
            else:
                # Create empty tensor with correct shape if no predictions
                all_predictions = torch.tensor(
                    [PADDING_VALUE] * 7,
                    device=self.device,
                )

            gathered_predictions = (
                self.gather(all_predictions, padding_value=PADDING_VALUE).detach().cpu()
            )

        return {
            "loss": loss,
            "logits": logits,
            "model_outputs": logits.view(B, S, V),
            "predictions": gathered_predictions,  # tensor of shape [N, 7]
            "pred_seq": pred_seq,
            "class_logits": class_logits,
            "batch_size": images.size(0),
        }

    def _format_predictions(
        self,
        boxes_list,
        labels_list,
        scores_list,
        image_ids,
        original_sizes,
        resized_sizes,
        predictions_list,
    ):
        """Format model predictions for evaluation."""
        for boxes, labels, scores, img_id, orig_size, res_size in zip(
            boxes_list,
            labels_list,
            scores_list,
            image_ids,
            original_sizes,
            resized_sizes,
        ):
            if scores is not None:
                if len(scores) != len(boxes):
                    warnings.warn(
                        f"Mismatched scores and boxes: {len(scores)} scores but {len(boxes)} boxes"
                    )
                    continue

                scaled_boxes = scale_bboxes_to_original_image_size(
                    boxes, res_size, orig_size, is_padded=True
                )

                # Stack predictions in expected format
                predictions = torch.cat(
                    [
                        scaled_boxes,  # [N, 4] - xmin, ymin, xmax, ymax
                        scores.unsqueeze(-1),  # [N, 1] - score
                        labels.unsqueeze(-1),  # [N, 1] - class_id
                        torch.full(
                            (len(boxes), 1),
                            img_id.item(),
                            device=boxes.device,
                        ),  # [N, 1] - image_id
                    ],
                    dim=1,
                )  # [N, 7]

                predictions_list.append(predictions)
