"""mAP evaluation for RL training, reusing the base training evaluation path.

Box scaling goes through ``training.trainer.format_predictions_for_evaluation``
(the same code used by ``Pix2SeqTrainer``), which expects normalized [0,1] XYXY
boxes and performs the single denormalize-and-unpad step. Class ids are
converted COCO-80 -> COCO-91 exactly like the base evaluation callback.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader

from data.base_dataset import coco80_to_coco91_lookup
from evaluation.coco_evaluator import COCOMeanAveragePrecision
from training.trainer import format_predictions_for_evaluation


def evaluate_map(
    model,
    eval_dataset,
    collate_fn,
    token_processor,
    val_json: dict,
    device: torch.device,
    max_seq_len: int,
    batch_size: int = 16,
    confidence_threshold: float = 0.05,
    accelerator=None,
    max_batches: Optional[int] = None,
) -> float:
    """Evaluate mAP with greedy decoding.

    Runs on a single process; when using distributed training call this on the
    main process only (and synchronise around it).

    Args:
        model: Pix2Seq model (wrapped or unwrapped)
        eval_dataset: Evaluation dataset
        collate_fn: Collator producing image/image_id/orig_image_sizes keys
        token_processor: TokenProcessor for decoding
        val_json: COCO validation annotations (targets)
        device: Device to run evaluation on
        max_seq_len: Maximum sequence length for generation
        batch_size: Evaluation batch size
        confidence_threshold: Minimum confidence for keeping predictions
            (matches base training evaluation, default 0.05)
        accelerator: Optional Accelerator (used only to unwrap the model)
        max_batches: Optional cap on number of batches (for smoke tests)

    Returns:
        mAP value (or -1 if no predictions were made)
    """
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
    else:
        unwrapped_model = model

    was_training = unwrapped_model.training
    unwrapped_model.eval()

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    coco80_to_91 = coco80_to_coco91_lookup()
    predictions_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            orig_sizes = batch["orig_image_sizes"].to(device)

            sequences, class_logits, _ = unwrapped_model.infer(
                images=images,
                max_seq_len=max_seq_len,
                greedy=True,
            )

            boxes_list, labels_list, scores_list = (
                token_processor.post_process_sequences(
                    sequences=sequences,
                    class_logits=class_logits,
                    confidence_threshold=confidence_threshold,
                )
            )

            resized_sizes = torch.as_tensor(
                images.shape[2:], device=device
            )[None].repeat(len(scores_list), 1)

            # Reuses the base-training scaling: boxes stay normalized until the
            # single scale_bboxes_to_original_image_size call inside
            format_predictions_for_evaluation(
                boxes_list,
                labels_list,
                scores_list,
                image_ids,
                orig_sizes,
                resized_sizes,
                predictions_list,
            )

    if was_training:
        unwrapped_model.train()

    coco_predictions = []
    for preds in predictions_list:
        for row in preds:
            x1, y1, x2, y2, score, class_id, image_id = row.tolist()
            coco_predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": coco80_to_91[int(class_id)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )

    evaluator = COCOMeanAveragePrecision(verbose=False)
    return evaluator.compute(val_json, coco_predictions)
