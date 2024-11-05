from __future__ import annotations

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.widgets import PolygonSelector
from scipy import ndimage
from shapely.geometry import Polygon
from skimage import measure
from skimage.filters import gaussian, threshold_otsu
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.segmentation import expand_labels
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.transformations import get_transformation

from .._constants import ROI
from .._sdata import get_spatial_image
from .image import resize

log = logging.getLogger(__name__)

HELPER = """Enclose cells within a polygon. Helper:
    - Click on the plot to add a polygon vertex
    - Press the 'esc' key to start a new polygon
    - Try holding the 'ctrl' key to move a single vertex
    - Once the polygon is finished and overlaid in red, you can close the window
"""

VALID_N_CHANNELS = [1, 3]

REGION_PROPERTIES = {
    "label": 0,
    "area": 2,
    "axis_major_length": 1,
    "axis_minor_length": 1,
    "bbox": 1,
    "centroid": 1,
    "eccentricity": 0,
    "equivalent_diameter_area": 1,
}


def _prepare(
    sdata: SpatialData,
    channels: list[str] | str | None,
    scale_factor: float,
    aggregate_channels: bool = False,
    image_key: str | None = None,
):
    image_key, spatial_image = get_spatial_image(sdata, return_key=True, key=image_key)
    image = spatial_image.transpose("y", "x", "c")

    if isinstance(channels, str):
        channels = [channels]

    if aggregate_channels:
        if channels is not None and len(channels):
            image = image.sel(c=channels)
            image = image.max(axis=-1)
        else:
            image = image.max(axis=-1)
    else:
        if channels is not None and len(channels):
            assert (
                len(channels) in VALID_N_CHANNELS
            ), f"Number of channels provided must be in: {', '.join(VALID_N_CHANNELS)}"
            image = image.sel(c=channels)
        else:
            assert (
                len(image.coords["c"]) in VALID_N_CHANNELS
            ), f"Choose one or three channels among {image.c.values} by using the --channels argument"

    log.info(f"Resizing image by a factor of {scale_factor}")
    return image_key, resize(image, scale_factor).compute()


def _compute_transcript_density(sdata: SpatialData, filter_key: str = "Blank"):
    geo_df = sdata[ROI.KEY]
    transcripts_key = next(iter(sdata.points))

    sdata_agg = sdata.aggregate(values=transcripts_key, by=ROI.KEY, agg_func="count")
    adata = sdata_agg["table"][:, ~sdata_agg["table"].var_names.str.startswith(filter_key)].copy()

    geo_df["n_transcripts"] = np.array(adata.X.sum(1)).flatten().copy()
    geo_df["transcript_density"] = geo_df["n_transcripts"] / geo_df["area"]

    return geo_df


def _filter_rois(geo_df: gpd.GeoDataFrame, area_threshold: float | None, density_threshold: float | None):

    if area_threshold is not None:
        mask = geo_df["area"] > area_threshold
        if np.sum(~mask) > 0:
            filter_areas = list(geo_df.loc[~mask, "area"].values)
            log.info(
                f"Filtering {np.sum(~mask):,} regions with area < {area_threshold:,} ({', '.join(f'{area:,}' for area in filter_areas)})"
            )
        geo_df = geo_df[mask]

    if density_threshold is not None:
        mask = geo_df["transcript_density"] > density_threshold
        if np.sum(~mask) > 0:
            filter_densities = list(geo_df.loc[~mask, "transcript_density"].values)
            log.info(
                f"Filtering {np.sum(~mask)} regions with density < {density_threshold} ({', '.join(map(str, filter_densities))})"
            )
        geo_df = geo_df[mask]

    return geo_df


def _process_image(image: np.ndarray, sigma: float, disk_size: float, expand: int):

    # Apply gaussian smoothing and thresholding
    image = gaussian(image, sigma=sigma)
    thr = threshold_otsu(image)
    image = (image > thr).astype(int)

    # Fill holes in binary image and apply median filter to remove outliers
    image = ndimage.binary_fill_holes(image).astype("uint8")
    image = median(image, disk(disk_size))

    # Label connected components (regions) and expand the regions
    image = measure.label(image)
    image = expand_labels(image, distance=expand)

    # Fill holes and label again to take care of slightly detached tissue
    image = ndimage.binary_fill_holes(image).astype("uint8")
    image = measure.label(image)

    return image


def _polygon_extractor(row, bbox: bool, scale_factor: float, image: np.ndarray):
    if bbox:
        return _create_bbox_polygon(row, scale_factor)
    else:
        return _create_contour_polygon(row, scale_factor, image)


def _create_bbox_polygon(row, scale_factor: float) -> Polygon:
    return Polygon(
        [
            (row["bbox-1"] * scale_factor, row["bbox-0"] * scale_factor),
            (row["bbox-1"] * scale_factor, row["bbox-2"] * scale_factor),
            (row["bbox-3"] * scale_factor, row["bbox-2"] * scale_factor),
            (row["bbox-3"] * scale_factor, row["bbox-0"] * scale_factor),
        ]
    )


def _create_contour_polygon(row, scale_factor: float, image: np.ndarray, n_pixel_pad: int = 1) -> Polygon:
    # pad the image to avoid boundary issues
    padded_image = np.pad(
        image,
        pad_width=n_pixel_pad,
        mode="constant",
        constant_values=0,
    )

    # Find contours for the region
    contours = measure.find_contours(padded_image == row["label"], 0.5)

    # Raise an error if more than one contour is found for a region
    if len(contours) != 1:
        raise ValueError(f"More than one contour found for region {row['label']}")

    # Raise an error if the countour is not closed
    if not np.allclose(contours[0][0], contours[0][-1]):
        raise ValueError(f"Contour for region {row['label']} is not closed")

    scaled_contour = ((contours[0] - n_pixel_pad / 2) * scale_factor)[:, ::-1]

    return Polygon(scaled_contour)


def _convert_scales(geo_df, scale_factor):
    for col in geo_df.columns:
        key = col.split("-")[0]
        if key in REGION_PROPERTIES.keys():
            if REGION_PROPERTIES[key] == 1:
                geo_df[col] = geo_df[col] * scale_factor
            elif REGION_PROPERTIES[key] == 2:
                geo_df[col] = geo_df[col] * scale_factor**2


def _identify_rois(
    img: np.array,
    scale_factor: float,
    sigma: float,
    disk_size: float,
    expand: int,
    bbox: bool,
) -> gpd.GeoDataFrame:

    # convert parameters given the `scale_factor`:
    sigma = sigma / scale_factor
    disk_size = int(disk_size / scale_factor)
    expand = int(expand / scale_factor)

    # Process the image
    img = _process_image(img, sigma, disk_size, expand)

    # Extract region properties
    region_dict = measure.regionprops_table(img, properties=REGION_PROPERTIES.keys())

    if len(region_dict) > 0:
        region_df = pd.DataFrame(region_dict)
        region_df["label"] = region_df["label"].astype("category")

        # Create polygond and filter out invalid ones
        # region_df["geometry"] = region_df.apply(polygon_lambda, axis=1)
        region_df["geometry"] = region_df.apply(lambda row: _polygon_extractor(row, bbox, scale_factor, img), axis=1)
        region_df = region_df[region_df["geometry"].apply(lambda p: p.is_valid)]

        geo_df = gpd.GeoDataFrame(region_df)

    else:
        log.warning("No region found. Using the whole image as the bounding box.")
        geo_df = _get_entire_image_bbox(img, scale_factor)

    # convert the measured properties back to the original scale
    _convert_scales(geo_df, scale_factor)

    return geo_df


def _get_entire_image_bbox(image, scale_factor):
    geo_df = gpd.GeoDataFrame(
        {
            "area": [image.shape[0] * image.shape[1] * scale_factor**2],
            "geometry": [
                Polygon(
                    [
                        (0, 0),
                        (0, image.shape[0] * scale_factor),
                        (image.shape[1] * scale_factor, image.shape[0] * scale_factor),
                        (image.shape[1] * scale_factor, 0),
                    ]
                )
            ],
        }
    )

    return geo_df


class _Selector:
    def __init__(self, ax):
        self.poly = PolygonSelector(ax, self.onselect, draw_bounding_box=True)
        log.info(HELPER)
        plt.show()

    def onselect(self, vertices):
        self.vertices = np.array(vertices)
        log.info(f"Selected polygon with {len(self.vertices)} vertices.")

    def disconnect(self):
        self.poly.disconnect_events()


def _draw_polygon(image: np.ndarray, scale_factor: float, margin_ratio: float):
    _, ax = plt.subplots()
    ax.imshow(image)

    dy, dx, *_ = image.shape
    plt.xlim(-margin_ratio * dx, dx + margin_ratio * dx)
    plt.ylim(dy + margin_ratio * dy, -margin_ratio * dy)

    selector = _Selector(ax)

    return Polygon(selector.vertices * scale_factor)


def intermediate_selection(intermediate_image: str, intermediate_polygon: str, margin_ratio: float = 0.1):
    log.info(f"Reading intermediate image {intermediate_image}")

    z = zarr.open(intermediate_image, mode="r")
    image = z[ROI.IMAGE_ARRAY_KEY][:]

    polygon = _draw_polygon(image, z.attrs[ROI.SCALE_FACTOR], margin_ratio)

    with zarr.ZipStore(intermediate_polygon, mode="w") as store:
        g = zarr.group(store=store)
        g.attrs.put({ROI.IMAGE_KEY: z.attrs[ROI.IMAGE_KEY], ROI.ELEMENT_TYPE: "images"})

        coords = np.array(polygon.exterior.coords)
        g.array(ROI.POLYGON_ARRAY_KEY, coords, dtype=coords.dtype, chunks=coords.shape)


def polygon_selection(
    sdata: SpatialData,
    intermediate_image: str | None = None,
    intermediate_polygon: str | None = None,
    channels: list[str] | None = None,
    scale_factor: float = 10,
    margin_ratio: float = 0.1,
):
    """Crop an image based on a user-defined polygon (interactive mode).

    Args:
        sdata: A `SpatialData` object
        intermediate_image: Path to the intermediate image, with a `.zip` extension. Use this only if the interactive mode is not available
        intermediate_polygon: Path to the intermediate polygon, with a `.zip` extension. Use this locally, after downloading the `intermediate_image`
        channels: List of channel names to be displayed. Optional if there are already only 1 or 3 channels.
        scale_factor: Resize the image by this value (high value for a lower memory usage)
        margin_ratio: Ratio of the image margin on the display (compared to the image size)
    """
    if intermediate_polygon is None:
        image_key, image = _prepare(sdata, channels, scale_factor)

        if intermediate_image is not None:
            log.info(f"Resized image will be saved to {intermediate_image}")
            with zarr.ZipStore(intermediate_image, mode="w") as store:
                g = zarr.group(store=store)
                g.attrs.put({ROI.SCALE_FACTOR: scale_factor, ROI.IMAGE_KEY: image_key})
                g.array(ROI.IMAGE_ARRAY_KEY, image, dtype=image.dtype, chunks=image.shape)
            return

        polygon = _draw_polygon(image, scale_factor, margin_ratio)
    else:
        log.info(f"Reading polygon at path {intermediate_polygon}")
        z = zarr.open(intermediate_polygon, mode="r")
        polygon = Polygon(z[ROI.POLYGON_ARRAY_KEY][:])
        image_key = z.attrs[ROI.IMAGE_KEY]

        image = get_spatial_image(sdata, image_key)

    geo_df = gpd.GeoDataFrame(geometry=[polygon])

    geo_df = ShapesModel.parse(geo_df, transformations=get_transformation(sdata[image_key], get_all=True).copy())
    sdata.shapes[ROI.KEY] = geo_df
    if sdata.is_backed():
        sdata.write_element(ROI.KEY, overwrite=True)

    log.info(f"Polygon saved in sdata['{ROI.KEY}']")


def automatic_polygon_selection(
    sdata: SpatialData,
    scale_factor: float = 10,
    channels: list[str] | None = None,
    sigma: float = 120,
    expand: int = 240,
    disk_size: int = 240,
    area_threshold: float = 300000,
    density_threshold: float = 1e-3,
    bbox: bool = False,
    image_key: str | None = None,
    write_element: bool = True,
):
    """Automatically identify a rectangular region of interest.

    Works well for images which have a clear background and foreground, e.g. organoids.

    Args:
        sdata: A `SpatialData` object
        scale_factor: Resize the image by this value (high value for a lower memory usage)
        channels: List of channel names to be used. These will be aggregated.
        sigma: Standard deviation for gaussian smoothing
        expand: Distance to expand the regions
        disk_size: Radius of the disk used in median filtering
        area_threshold: Minimum size of regions
        bbox: If True, the combined bounding box of the regions will be used as the polygon
        image_key: Key of the image to be used. None is only allowed if there is a single image in the sdata object.
        write_element: If True, the polygon will be written to disk.

    """

    image_key, image = _prepare(
        sdata, channels=channels, scale_factor=scale_factor, image_key=image_key, aggregate_channels=True
    )

    # compute the bounding box for all regions
    geo_df = _identify_rois(
        image,
        scale_factor=scale_factor,
        sigma=sigma,
        disk_size=disk_size,
        expand=expand,
        bbox=bbox,
    )

    sdata.shapes[ROI.KEY] = ShapesModel.parse(
        geo_df, transformations=get_transformation(sdata[image_key], get_all=True).copy()
    )

    # compute transcript density under each ROI and filter them
    geo_df = _compute_transcript_density(sdata)
    geo_df = _filter_rois(geo_df, area_threshold=area_threshold, density_threshold=density_threshold)

    if geo_df.shape[0] == 0:
        log.warning("No region found. Using the whole image as the bounding box.")
        geo_df = _get_entire_image_bbox(image, scale_factor)
        _convert_scales(geo_df, scale_factor)

    sdata.shapes[ROI.KEY] = ShapesModel.parse(geo_df)

    if sdata.is_backed() and write_element:
        sdata.write_element(ROI.KEY, overwrite=True)

    log.info(f"Polygon saved in sdata['{ROI.KEY}']")
