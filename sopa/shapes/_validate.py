import logging

import geopandas as gpd
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

log = logging.getLogger(__name__)


def to_valid_polygons(geo_df: gpd.GeoDataFrame, simple_polygon: bool = True) -> gpd.GeoDataFrame:
    geo_df.geometry = geo_df.geometry.map(lambda cell: ensure_polygon(cell, simple_polygon))
    return geo_df[~geo_df.is_empty]


def ensure_polygon(
    cell: Polygon | MultiPolygon | GeometryCollection, simple_polygon: bool = True
) -> Polygon | MultiPolygon:
    """Ensures that the provided cell becomes a Polygon

    Args:
        cell: A shapely Polygon or MultiPolygon or GeometryCollection
        simple_polygon: If True, will return a Polygon without holes. Else, allow holes and MultiPolygon.

    Returns:
        The shape as a Polygon, or an empty Polygon if the cell was invalid
    """
    cell = shapely.make_valid(cell)

    if isinstance(cell, Polygon):
        if simple_polygon and cell.interiors:
            cell = Polygon(cell.exterior)
        return cell

    if isinstance(cell, MultiPolygon):
        return max(cell.geoms, key=lambda polygon: polygon.area) if simple_polygon else cell

    if isinstance(cell, GeometryCollection):
        geoms = [geom for geom in cell.geoms if isinstance(geom, Polygon)]

        if len(geoms) > 1 and not simple_polygon:
            return MultiPolygon(geoms)

        if geoms:
            return max(geoms, key=lambda polygon: polygon.area)

        log.warning(f"Removing cell of type {type(cell)} as it contains no Polygon geometry")
        return Polygon()

    log.warning(f"Removing cell of unknown type {type(cell)}")
    return Polygon()


def _smoothen_cell(
    cell: MultiPolygon,
    smooth_radius: float,
    tolerance: float,
    debug_save_path: str | None = None,
    cells_gdf: gpd.GeoDataFrame | None = None,
) -> Polygon:
    """Smoothen a cell polygon

    Args:
        cell: MultiPolygon representing a cell
        smooth_radius: Radius used to smooth the cell polygon
        tolerance: Tolerance used to simplify the cell polygon
        debug_save_path: Optional path to save the GeoDataFrame when topology exceptions occur
        cells_gdf: Optional GeoDataFrame to save for debugging

    Returns:
        Shapely polygon representing the cell, or an empty Polygon if the cell was empty after smoothing
    """
    from shapely.errors import GEOSException

    try:
        cell = cell.buffer(-smooth_radius).buffer(2 * smooth_radius).buffer(-smooth_radius)
        cell = cell.simplify(tolerance)
        return ensure_polygon(cell)
    except GEOSException:
        # Save debug data if path is provided
        if debug_save_path is not None and cells_gdf is not None:
            try:
                cells_gdf.to_file(debug_save_path)
                log.warning(
                    "Topology exception occurred during cell smoothing. GeoDataFrame saved to: %s", debug_save_path
                )
            except (OSError, ValueError) as save_error:
                log.warning("Failed to save debug GeoDataFrame to %s: %s", debug_save_path, save_error)

        # Handle topology exceptions by returning the original cell with minimal smoothing
        log.warning("Cell smoothing failed with GEOSException, trying reduced smooth radius")
        try:
            # Try with a smaller smooth radius
            reduced_smooth_radius = smooth_radius * 0.1
            cell = cell.buffer(-reduced_smooth_radius).buffer(2 * reduced_smooth_radius).buffer(-reduced_smooth_radius)
            cell = cell.simplify(tolerance)
            return ensure_polygon(cell)
        except GEOSException:
            # If still failing, just simplify without buffering
            log.warning("Reduced smooth radius failed, attempting simplification only")
            try:
                return ensure_polygon(cell.simplify(tolerance))
            except GEOSException:
                # Return empty polygon as last resort
                log.warning("All smoothing attempts failed, returning empty polygon")
                return Polygon()


def _default_tolerance(mean_radius: float) -> float:
    if mean_radius < 10:
        return 0.4
    if mean_radius < 20:
        return 1
    return 2
