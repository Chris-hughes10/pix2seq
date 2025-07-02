import datetime
import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from evaluation.calculate_map_callback import (
    CalculateMeanAveragePrecisionCallback,
    ConvertPredictionClassesCallback,
)
from func_to_script import load_config_from_yaml, script
from model.model import Pix2SeqModel
from model.modelv2 import LlamaPix2Seq
from pytorch_accelerated.callbacks import (
    SaveBestModelCallback,
    WSDCheckpointCallback,
    get_default_callbacks,
)
from pytorch_accelerated.schedulers import CosineLrScheduler
from pytorch_accelerated.schedulers.wsd_scheduler import WSDLrScheduler
from training.callbacks import (
    AzureMLLoggerCallback,
    LearningRateTrackerCallback,
    TokenAccuracyCallback,
    VisualizePredictionsCallback,
)
from training.trainer import Pix2SeqTrainer

from data.base_dataset import COCOBaseDataset
from data.dataset import Pix2SeqCollator, Pix2SeqDataset
from data.tokenizer import LabelCorruptionStrategy, TokenProcessor


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

    train_dataset = Pix2SeqDataset(
        base_dataset=train_ds,
        num_classes=config.data.num_classes,
        training=True,
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

    # Create targets json for COCO evaluation
    # load json from config.data.val_annotation_file
    with open(val_annotation_file, "r") as f:
        val_json = json.load(f)

    return train_dataset, eval_dataset, val_json


def create_model_and_optimizer(
    config, token_processor: TokenProcessor, llama_model=False
):
    """Create model and optimizer."""

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.eps,
    )

    return model, optimizer


def setup_output_dir(config, output_dir, predictions_dir):
    """Create output directory and save config."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)


def copy_outputs(src_dir: Path, dest_dir: str = "./outputs"):
    """Copy all contents from source directory to destination directory."""

    dest_path = Path(dest_dir)
    dest_path.mkdir(exist_ok=True)

    # Copy the entire directory tree
    for item in src_dir.glob("*"):
        dest_item = dest_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item)

    print(f"Copied outputs from {src_dir} to {dest_dir}")


FILE_PATH = Path(__file__).resolve().parent


@script
def train(
    coco_dir: str = "/workspaces/object-detection-rl/data/coco",
    config_file: str = "overfit_eval.yaml",
    copy_output_dir: bool = False,
    use_progress_bar: bool = True,
):
    """Main training function with robust loss handling."""

    config = load_config_from_yaml((FILE_PATH / "config") / config_file)

    output_dir = Path(config.training.output_dir) / datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    predictions_dir = output_dir / config.training.predictions_dir

    setup_output_dir(config, output_dir, predictions_dir)
    coco_dir = Path(coco_dir)
    train_image_dir = coco_dir / "images/train2017"
    train_annotation_file = coco_dir / "annotations/instances_train2017.json"
    val_image_dir = coco_dir / "images/val2017"
    val_annotation_file = coco_dir / "annotations/instances_val2017.json"

    # Create datasets
    train_dataset, val_dataset, val_json = create_datasets(
        config,
        train_image_dir,
        train_annotation_file,
        val_image_dir,
        val_annotation_file,
    )

    # Calculate maximum sequence length to properly handle all steps
    boxes_with_eos_position = (
        train_dataset.max_instances + 1
    )  # Account for collator's EOS position
    tokens_from_boxes = boxes_with_eos_position * 5  # Each box becomes 5 tokens
    total_seq_len = tokens_from_boxes + 2  # Add BOS and EOS to

    token_processor = TokenProcessor(
        quantization_bins=config.tokenization.quantization_bins,
        noise_bbox_weight=config.tokenization.noise_bbox_weight,
        eos_token_weight=config.tokenization.eos_token_weight,
        max_seq_len=total_seq_len,
        num_classes=config.data.num_classes,
        num_special_tokens=10,
        corruption_strategy=LabelCorruptionStrategy(config.data.class_label_corruption),
    )

    # Create model and optimizer
    model, optimizer = create_model_and_optimizer(
        config, token_processor, llama_model=config.model.llama_model
    )

    if config.training.overfit_eval_set:
        # TokenAccuracyCallback depends on exact sequence ordering of tokens,
        # so it is only useful for debugging and not for real training.
        callbacks = [
            TokenAccuracyCallback(
                pad_token_id=token_processor.PADDING_TOKEN,
                eos_token_id=token_processor.EOS_TOKEN,
                print_samples=True,
            ),
        ]

    else:
        callbacks = []

    # Setup callbacks
    callbacks.extend(
        [
            ConvertPredictionClassesCallback,
            CalculateMeanAveragePrecisionCallback(
                targets_json=val_json,
                # iou_threshold=config.evaluation.iou_threshold,
                save_predictions_output_dir_path=output_dir / "predictions",
                verbose=False,
            ),
            VisualizePredictionsCallback(
                output_dir=output_dir / "visualizations",
                category_names=train_dataset.categories,
                save_freq=config.training.run_eval_freq,
                images_per_epoch=config.training.num_visualizations,
                confidence_threshold=0.1,
                figsize=(12, 6),  # Wider to accommodate side-by-side comparison
            ),
            *get_default_callbacks(progress_bar=use_progress_bar),
            LearningRateTrackerCallback,
            AzureMLLoggerCallback(),
            SaveBestModelCallback(
                watch_metric="map",
                greater_is_better=True,
                save_path=output_dir / "checkpoint.pt",
                load_saved_checkpoint=False,
            ),
        ]
    )

    if config.training.use_wsd_scheduler:
        callbacks.append(
            WSDCheckpointCallback(
                save_dir=output_dir,
                initial_checkpoint=(
                    config.model.pretrained_path
                    if config.model.pretrained_path
                    else None
                ),
            ),
        )

    if config.training.gradient_accumulation_steps > 1:
        print(
            f"Using gradient accumulation with {config.training.gradient_accumulation_steps} steps"
        )
        print(
            f"Effective batch size: {config.training.batch_size * config.training.gradient_accumulation_steps}"
        )

    # Create trainer
    trainer = Pix2SeqTrainer(
        model=model,
        optimizer=optimizer,
        callbacks=callbacks,
        run_eval_freq=config.training.run_eval_freq,
        # Generation params
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        temperature=config.generation.temperature,
        token_processor=token_processor,
        output_dir=predictions_dir,
    )

    if config.model.pretrained_path and not config.training.use_wsd_scheduler:
        trainer.load_checkpoint(
            config.model.pretrained_path, load_optimizer=True, load_scheduler=True
        )

    if config.training.use_wsd_scheduler:
        is_continuation_from_checkpoint = (
            True if config.model.pretrained_path else False
        )

        scheduler_fn = WSDLrScheduler.create_scheduler_fn(
            is_continuation_from_checkpoint=is_continuation_from_checkpoint,
            num_warmup_epochs=config.training.warmup_epochs,
        )
        print(
            f" Continuing training with WSD scheduler: {is_continuation_from_checkpoint}"
        )
    else:
        scheduler_fn = CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=config.training.warmup_epochs,
        )

    # Run training
    trainer.train(
        train_dataset=train_dataset
        if not config.training.overfit_eval_set
        else val_dataset,
        eval_dataset=val_dataset,
        num_epochs=config.training.num_epochs,
        per_device_batch_size=config.training.batch_size,
        max_num_train_steps=config.training.max_steps
        if config.training.max_steps
        else None,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_clip_value=config.training.gradient_clip_value,
        create_scheduler_fn=scheduler_fn,
        collate_fn=Pix2SeqCollator(
            token_processor, corrupt_and_randomise=config.tokenization.random_ordering
        ),
        train_dataloader_kwargs={
            "shuffle": not config.training.overfit_eval_set,
        },
        eval_dataloader_kwargs={
            "batch_size": config.training.eval_batch_size,
        },
    )

    # trainer.evaluate(val_dataset,
    #                  per_device_batch_size=config.training.eval_batch_size,
    #                 #  dataloader_kwargs={"num_workers": 0},
    #                  collate_fn=Pix2SeqCollator(token_processor))

    trainer.save_checkpoint(output_dir / "final_model.pt")

    # Avoids problems with AzureML accessing output files before they are uploaded
    if copy_output_dir:
        copy_outputs(output_dir)


if __name__ == "__main__":
    train()
