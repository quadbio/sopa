from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from .._constants import SopaKeys

log = logging.getLogger(__name__)


def cellpose_patch(
    diameter: float,
    channels: list[str],
    model_type: str = "cyto3",
    pretrained_model: str | bool = False,
    gpu: bool = False,
    cellpose_model_kwargs: dict | None = None,
    **cellpose_eval_kwargs: int,
) -> Callable:
    """Creation of a callable that runs Cellpose segmentation on a patch

    Args:
        diameter: Cellpose diameter parameter
        channels: List of channel names
        model_type: Cellpose model type
        pretrained_model: Path to the pretrained model to be loaded
        gpu: Whether to use the GPU
        cellpose_model_kwargs: Kwargs to be provided to the `cellpose.models.CellposeModel` object
        **cellpose_eval_kwargs: Kwargs to be provided to `model.eval` (where `model` is a `cellpose.models.CellposeModel` object)

    Returns:
        A `callable` whose input is an image of shape `(C, Y, X)` and output is a cell mask of shape `(Y, X)`. Each mask value `>0` represent a unique cell ID
    """
    try:
        from cellpose import models
    except ImportError:
        raise ImportError(
            "To use cellpose, you need its corresponding sopa extra: `pip install 'sopa[cellpose]'` (normal mode) or `pip install -e '.[cellpose]'` (if using snakemake)"
        )

    cellpose_model_kwargs = cellpose_model_kwargs or {}

    if pretrained_model:
        model = models.CellposeModel(pretrained_model=pretrained_model, gpu=gpu, **cellpose_model_kwargs)
    else:
        model = models.Cellpose(model_type=model_type, gpu=gpu, **cellpose_model_kwargs)

    log.info(f"Cellpose device: {model.device}")

    if isinstance(channels, str) or len(channels) == 1:
        channels = [0, 0]  # gray scale
    elif len(channels) == 2:
        channels = [1, 2]
    else:
        raise ValueError(f"Provide 1 or 2 channels. Found {len(channels)}")

    def _(patch: np.ndarray):
        mask, *_ = model.eval(patch, diameter=diameter, channels=channels, **cellpose_eval_kwargs)
        return mask

    return _


def dummy_method(**method_kwargs):
    """A method builder builder (i.e. it returns a segmentation function).
    Kwargs can be provided and used in the below function"""

    def segmentation_function(image: np.ndarray) -> np.ndarray:
        """A dummy example of a custom segmentation method
        that creates one cell (with a padding of 10 pixels).

        Args:
            image: An image of shape `(C, Y, X)`

        Returns:
            A mask of shape `(Y, X)` containing one cell
        """
        mask = np.zeros(image.shape[1:], dtype=int)

        # one cell, corresponding to value 1
        mask[10:-10, 10:-10] = 1  # squared shaped

        return mask

    return segmentation_function


def comseg_patch(temp_dir: str, patch_index: int, config: dict):
    import json

    try:
        import comseg
        from comseg import dataset as ds
        from comseg import dictionary
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Install comseg (`pip install comseg`) for this method to work")

    assert comseg.__version__ >= "1.3", "comseg version should be >= 1.3"

    path_dataset_folder = Path(temp_dir) / str(patch_index)

    dataset = ds.ComSegDataset(
        path_dataset_folder=path_dataset_folder,
        dict_scale=config["dict_scale"],
        mean_cell_diameter=config["mean_cell_diameter"],
        gene_column=config["gene_column"],
        image_csv_files=["transcripts.csv"],
        centroid_csv_files=["centroids.csv"],
        path_cell_centroid=path_dataset_folder,
        min_nb_rna_patch=config.get("min_nb_rna_patch", 0),
        prior_name=config.get("prior_name", SopaKeys.DEFAULT_CELL_KEY),
    )

    dataset.compute_edge_weight(config=config)

    Comsegdict = dictionary.ComSegDict(
        dataset=dataset,
        mean_cell_diameter=config["mean_cell_diameter"],
    )

    Comsegdict.run_all(config=config)

    if "return_polygon" in config:
        assert config["return_polygon"] is True, "Only return_polygon=True is supported in sopa"
    anndata_comseg, json_dict = Comsegdict.anndata_from_comseg_result(config=config)
    anndata_comseg.write_h5ad(path_dataset_folder / "segmentation_counts.h5ad")
    with open(path_dataset_folder / "segmentation_polygons.json", "w") as f:
        json.dump(json_dict["transcripts"], f)
