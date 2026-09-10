"""Admission for user-supplied site geometry; no invented basemap or spatial conclusions."""

from symplex.core.contracts import Invalid, number


def validate_geojson(data):
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise Invalid("Provide a GeoJSON FeatureCollection")
    features = data.get("features")
    if not isinstance(features, list) or not 1 <= len(features) <= 200:
        raise Invalid("Provide 1–200 site features")
    count = 0
    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise Invalid("Invalid GeoJSON feature")
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        if kind == "Point":
            coordinates = [geometry.get("coordinates")]
        elif kind == "Polygon":
            rings = geometry.get("coordinates")
            if (
                not isinstance(rings, list)
                or not rings
                or any(
                    not isinstance(r, list) or len(r) < 4 or r[0] != r[-1]
                    for r in rings
                )
            ):
                raise Invalid(
                    "Polygon rings must be closed with at least four coordinates"
                )
            coordinates = [p for ring in rings for p in ring]
        else:
            raise Invalid("Only Point and Polygon site features are supported")
        for coordinate in coordinates:
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise Invalid("Coordinates must be [longitude, latitude]")
            number(coordinate[0], -180, 180)
            number(coordinate[1], -90, 90)
            count += 1
            if count > 5000:
                raise Invalid("Geometry exceeds the 5000-vertex envelope")
        properties = feature.get("properties")
        if properties is not None and not isinstance(properties, dict):
            raise Invalid("Properties must be an object")
    return data
