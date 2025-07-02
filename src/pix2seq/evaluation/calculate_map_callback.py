import json
from pathlib import Path

import pandas as pd
import torch
from evaluation.coco_evaluator import (
    CLASS_ID_COL,
    IMAGE_ID_COL,
    SCORE_COL,
    XMAX_COL,
    XMIN_COL,
    YMAX_COL,
    YMIN_COL,
    COCOMeanAveragePrecision,
)
from pytorch_accelerated.callbacks import TrainerCallback

from data.base_dataset import coco80_to_coco91_lookup


class ConvertPredictionClassesCallback(TrainerCallback):
    """Callback to convert prediction class IDs from COCO-80 to COCO-91 format.

    The COCO dataset uses two different category ID systems:
    - COCO-80: A contiguous format with IDs 0-79 commonly used by detection models
    - COCO-91: The original format with IDs 1-91 (with some IDs skipped) used in evaluation

    This callback converts model predictions from COCO-80 to COCO-91 format during
    evaluation to ensure correct mean average precision (MAP) calculation.
    """

    def __init__(self):
        """Initialize the callback with COCO-80 to COCO-91 mapping."""
        self.lookup = coco80_to_coco91_lookup()

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        """Convert prediction class IDs after each evaluation step.

        Args:
            trainer: The Pix2SeqTrainer instance
            batch: Current batch of data
            batch_output: Dictionary containing model outputs including predictions
                The predictions tensor has shape [N, 7] with columns:
                [xmin, ymin, xmax, ymax, score, class_id, image_id]
            kwargs: Additional arguments passed by the trainer

        Note:
            Modifies batch_output["predictions"] in-place by converting the class_id
            column from COCO-80 to COCO-91 format.
        """
        predictions = batch_output["predictions"]
        # Check if predictions is empty or 1D
        if predictions.numel() == 0 or predictions.dim() == 1:
            return

        try:
            coco_80_class_ids = predictions[:, -2]
            coco_91_class_ids = torch.as_tensor(
                [self.lookup[int(c)] for c in coco_80_class_ids],
                device=predictions.device,
                dtype=predictions.dtype,
            )
            # modify batch output inplace
            batch_output["predictions"][:, -2] = coco_91_class_ids
        except Exception as e:
            print(f"Error processing predictions of shape {predictions.shape}: {e}")
            return


class CalculateMeanAveragePrecisionCallback(TrainerCallback):
    """
    A callback which accumulates predictions made during an epoch and uses these to calculate the Mean Average Precision
    from the given targets.

    .. Note:: If using distributed training or evaluation, this callback assumes that predictions have been gathered
    from all processes during the evaluation step of the main training loop.
    """

    def __init__(
        self,
        targets_json,
        iou_threshold=None,
        save_predictions_output_dir_path=None,
        verbose=False,
    ):
        """
        :param targets_json: a COCO-formatted dictionary with the keys "images", "categories" and "annotations"
        :param iou_threshold: If set, the IoU threshold at which mAP will be calculated. Otherwise, the COCO default range of IoU thresholds will be used.
        :param save_predictions_output_dir_path: If provided, the path to which the accumulated predictions will be saved, in coco json format.
        :param verbose: If True, display the output provided by pycocotools, containing the average precision and recall across a range of box sizes.
        """
        self.evaluator = COCOMeanAveragePrecision(iou_threshold)
        self.targets_json = targets_json
        self.verbose = verbose
        self.save_predictions_path = (
            Path(save_predictions_output_dir_path)
            if save_predictions_output_dir_path is not None
            else None
        )

        self.eval_predictions = []
        self.image_ids = set()

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        predictions = batch_output["predictions"]
        if len(predictions) > 0:
            self._update(predictions)

    def on_eval_epoch_end(self, trainer, **kwargs):
        preds_df = pd.DataFrame(
            self.eval_predictions,
            columns=[
                XMIN_COL,
                YMIN_COL,
                XMAX_COL,
                YMAX_COL,
                SCORE_COL,
                CLASS_ID_COL,
                IMAGE_ID_COL,
            ],
        )

        predictions_json = self.evaluator.create_predictions_coco_json_from_df(preds_df)
        self._save_predictions(trainer, predictions_json, preds_df)

        if self.verbose and trainer.run_config.is_local_process_zero:
            self.evaluator.verbose = True

        map_ = self.evaluator.compute(self.targets_json, predictions_json)
        trainer.run_history.update_metric("map", map_)

        self._reset()

    @classmethod
    def create_from_targets_df(
        cls,
        targets_df,
        image_ids,
        iou_threshold=None,
        save_predictions_output_dir_path=None,
        verbose=False,
    ):
        """
        Create an instance of :class:`CalculateMeanAveragePrecisionCallback` from a dataframe containing the ground
        truth targets and a collections of all image ids in the dataset.

        :param targets_df: DF w/ cols: ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
        :param image_ids: A collection of all image ids in the dataset, including those without annotations.
        :param iou_threshold:  If set, the IoU threshold at which mAP will be calculated. Otherwise, the COCO default range of IoU thresholds will be used.
        :param save_predictions_output_dir_path: If provided, the path to which the accumulated predictions will be saved, in coco json format.
        :param verbose: If True, display the output provided by pycocotools, containing the average precision and recall across a range of box sizes.
        :return: An instance of :class:`CalculateMeanAveragePrecisionCallback`
        """

        targets_json = COCOMeanAveragePrecision.create_targets_coco_json_from_df(
            targets_df, image_ids
        )

        return cls(
            targets_json=targets_json,
            iou_threshold=iou_threshold,
            save_predictions_output_dir_path=save_predictions_output_dir_path,
            verbose=verbose,
        )

    def _remove_seen(self, labels):
        """
        Remove any image id that has already been seen during the evaluation epoch. This can arise when performing
        distributed evaluation on a dataset where the batch size does not evenly divide the number of samples.

        """
        image_ids = labels[:, -1].tolist()

        # remove any image_idx that has already been seen
        # this can arise from distributed training where batch size does not evenly divide dataset
        seen_id_mask = torch.as_tensor(
            [False if idx not in self.image_ids else True for idx in image_ids]
        )

        if seen_id_mask.all():
            # no update required as all ids already seen this pass
            return []
        elif seen_id_mask.any():  # at least one True
            # remove predictions for images already seen this pass
            labels = labels[~seen_id_mask]

        return labels

    def _update(self, predictions):
        filtered_predictions = self._remove_seen(predictions)

        if len(filtered_predictions) > 0:
            self.eval_predictions.extend(filtered_predictions.tolist())
            updated_ids = filtered_predictions[:, -1].unique().tolist()
            self.image_ids.update(updated_ids)

    def _reset(self):
        self.image_ids = set()
        self.eval_predictions = []

    def _save_predictions(self, trainer, predictions_json, preds_df):
        if (
            self.save_predictions_path is not None
            and trainer.run_config.is_world_process_zero
        ):
            if len(predictions_json) > 0:
                with open(self.save_predictions_path / "predictions.json", "w") as f:
                    json.dump(predictions_json, f)
                preds_df.to_csv(
                    self.save_predictions_path / "predictions.csv", index=False
                )
