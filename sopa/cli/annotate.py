import ast

import typer

app_annotate = typer.Typer()
option = typer.Option()


@app_annotate.command()
def fluorescence(
    sdata_path: str,
    marker_cell_dict: str = typer.Option(default={}, callback=ast.literal_eval),
    cell_type_key: str = "cell_type",
):
    """Simple annotation based on fluorescence, where each one channel corresponds to one cell type.

    Args:\n
        sdata_path: Path to the SpatialData zarr directory\n
        marker_cell_dict: Dictionary chose keys are channel names, and values are the corresponding cell types\n
        cell_type_key: Key added in 'adata.obs' corresponding to the cell type\n
    """
    from pathlib import Path

    from sopa.annotation.fluorescence import higher_z_score
    from sopa.io.standardize import read_zarr_standardized

    sdata = read_zarr_standardized(sdata_path)

    assert sdata.table is not None, f"Annotation requires `sdata.table` to be not None"

    higher_z_score(sdata.table, marker_cell_dict, cell_type_key)
    sdata.table.write_zarr(Path(sdata_path) / "table" / "table")


@app_annotate.command()
def tangram(
    sdata_path: str,
    sc_reference_path: str = option,
    cell_type_key: str = "cell_type",
    reference_preprocessing: str = None,
    bag_size: int = 10_000,
    max_obs_reference: int = 10_000,
):
    """Tangram segmentation (i.e., uses an annotated scRNAseq reference to transfer cell-types)

    Args:\n
        sdata_path: Path to the SpatialData zarr directory\n
        sc_reference_path: Path to the scRNAseq annotated reference\n
        cell_type_key: Key of 'adata_ref.obs' containing the cell-types\n
        reference_preprocessing: Preprocessing method applied to the reference. Either None (raw counts), or 'normalized' (sc.pp.normalize_total) or 'log1p' (sc.pp.normalize_total and sc.pp.log1p)\n
        bag_size: Number of cells in each bag of the spatial table. Low values will decrease the memory usage\n
        max_obs_reference: Maximum samples to be considered in the reference for tangram. Low values will decrease the memory usage\n
    """
    from pathlib import Path

    import anndata

    from sopa.annotation.tangram.run import tangram_annotate
    from sopa.io.standardize import read_zarr_standardized

    sdata = read_zarr_standardized(sdata_path)
    adata_sc = anndata.read(sc_reference_path)

    tangram_annotate(
        sdata,
        adata_sc,
        cell_type_key,
        reference_preprocessing=reference_preprocessing,
        bag_size=bag_size,
        max_obs_reference=max_obs_reference,
    )
    sdata.table.write_zarr(Path(sdata_path) / "table" / "table")
