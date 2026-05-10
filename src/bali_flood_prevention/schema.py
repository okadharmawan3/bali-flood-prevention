"""Flood label schema and validation helpers."""

import json
from collections import OrderedDict
from typing import Any

RISK_LEVELS = ("low", "medium", "high")
WATER_EXTENT_LEVELS = ("small", "moderate", "large")
CONFIDENCE_LEVELS = ("low", "medium", "high")

LABEL_FIELDS = (
    "flood_risk_level",
    "water_extent_level",
    "standing_water_present",
    "temporary_inundation_likely",
    "urban_or_infrastructure_exposure",
    "road_or_transport_disruption_likely",
    "cropland_or_settlement_exposure",
    "river_or_coastal_overflow_context",
    "low_lying_or_poor_drainage_area",
    "vegetation_or_soil_saturation",
    "permanent_water_body_present",
    "cloud_shadow_or_image_quality_limited",
    "confidence",
)

BOOLEAN_FIELDS = tuple(
    field
    for field in LABEL_FIELDS
    if field
    not in {
        "flood_risk_level",
        "water_extent_level",
        "confidence",
    }
)

SYSTEM_PROMPT = """\
You are a remote sensing analyst specialising in flood-risk assessment for Bali,
Indonesia. You will be given two Sentinel-2 satellite images of the same land
tile:
  1. RGB composite (bands B4-B3-B2): natural colour, useful for settlements,
     roads, rivers, coastlines, cropland, cloud, terrain texture, and visible
     standing water.
  2. SWIR composite (bands B12-B8-B4): useful for separating water, wet soil,
     saturated vegetation, and dry land. Open water is often very dark in SWIR;
     wet soil or saturated vegetation may appear darker/muted relative to dry
     vegetation; built-up land and bare soil can appear bright or pink/orange.

Assess the flood-prevention risk of the tile and return ONLY a valid JSON object
with exactly the required fields. Do not include markdown, captions, comments,
or explanation outside the JSON.
"""

USER_TEXT = (
    "Image 1 is the RGB composite. Image 2 is the SWIR composite. "
    "Return the Bali flood-risk JSON for this tile."
)

FLOOD_LABEL_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "flood_risk_level": {
            "type": "string",
            "enum": list(RISK_LEVELS),
            "description": "Overall priority for flood-prevention response.",
        },
        "water_extent_level": {
            "type": "string",
            "enum": list(WATER_EXTENT_LEVELS),
            "description": "Approximate visible floodwater or surface-water coverage.",
        },
        "standing_water_present": {"type": "boolean"},
        "temporary_inundation_likely": {"type": "boolean"},
        "urban_or_infrastructure_exposure": {"type": "boolean"},
        "road_or_transport_disruption_likely": {"type": "boolean"},
        "cropland_or_settlement_exposure": {"type": "boolean"},
        "river_or_coastal_overflow_context": {"type": "boolean"},
        "low_lying_or_poor_drainage_area": {"type": "boolean"},
        "vegetation_or_soil_saturation": {"type": "boolean"},
        "permanent_water_body_present": {"type": "boolean"},
        "cloud_shadow_or_image_quality_limited": {"type": "boolean"},
        "confidence": {
            "type": "string",
            "enum": list(CONFIDENCE_LEVELS),
            "description": "Confidence in the complete label set.",
        },
    },
    "required": list(LABEL_FIELDS),
    "additionalProperties": False,
}

OPENAI_LABELING_INSTRUCTIONS = """\
You are a remote sensing analyst labeling Sentinel-2 image pairs for a Bali,
Indonesia flood-prevention dataset.

Inputs:
- Image 1 is RGB natural color (B4-B3-B2).
- Image 2 is SWIR/NIR/red (B12-B8-B4).
- Metadata provides region, timestamp, tile center, point id, and tile size.

Output contract:
Return exactly one JSON object matching the schema. Do not include markdown,
captions, comments, explanations, or extra fields.

Professional interpretation workflow:
1. First assess image quality. Identify cloud, cloud shadow, haze, glare,
   no-data, or low-contrast areas. If these obscure important land/water
   features, set cloud_shadow_or_image_quality_limited=true and reduce
   confidence.
2. Identify permanent water and drainage context: ocean/coastline, rivers,
   canals, reservoirs, ponds, wetlands, fishponds, estuaries, deltas, and
   drainage channels. Permanent water is valid flood context, not necessarily
   active flooding.
3. Identify exposure context: dense settlement, roads, bridges, industrial
   areas, ports, airports, utilities, rural settlements, cropland, paddy fields,
   plantations, and peri-urban agriculture. These fields can be true even when
   no active flooding is visible.
4. Identify active flood evidence: abnormal standing water, water outside
   normal channels, expanded floodplain/coastal margin water, water across
   fields/urban blocks/roads, or SWIR-dark wet surfaces that are spatially
   inconsistent with normal permanent water.
5. Assign risk_level by combining active flood evidence, water extent, exposure,
   terrain/drainage context, and image quality.

Field definitions and criteria:
- flood_risk_level:
    - low: no active flood evidence. The tile may still contain urban areas,
      cropland, rivers, coastline, or other exposure/context features.
    - medium: possible localized standing water/saturation, flood-prone context
      with exposed settlement/cropland/transport, or ambiguous water expansion.
    - high: visible abnormal inundation, large water extent, or likely
      floodwater affecting or directly threatening dense settlement, roads,
      bridges, infrastructure, cropland, or rural settlements.
- water_extent_level:
    - small: no visible water or only narrow/small water features.
    - moderate: visible water/wet surfaces cover a meaningful but not dominant
      part of the tile.
    - large: water/wet surfaces dominate the tile or cover broad floodplain,
      coastal, urban, or agricultural areas.
- standing_water_present: true when visible water is present, including normal
  ocean, rivers, reservoirs, ponds, fishponds, wetlands, canals, or floodwater.
  Use false only when no visible standing water can be identified.
- temporary_inundation_likely: true only when the water pattern appears abnormal
  or temporary, such as water outside normal channels, across cropland or urban
  blocks, over roads, or expanded from river/coastal margins. Normal ocean,
  reservoirs, ponds, and stable rivers alone are not temporary inundation.
- urban_or_infrastructure_exposure: true when buildings, dense settlements,
  roads, bridges, industrial/commercial zones, ports, airports, power/utility
  facilities, or other infrastructure are visible or adjacent to flood-relevant
  drainage/water context. This does not require visible flooding.
- road_or_transport_disruption_likely: true only when roads, bridges, railways,
  or transport corridors appear flooded, cut off, crossed by likely floodwater,
  or immediately threatened by adjacent abnormal inundation.
- cropland_or_settlement_exposure: true when cropland, paddy fields,
  plantations, rural settlements, peri-urban agriculture, or mixed agricultural
  settlement areas are visible and could plausibly be exposed to floodwater.
  This does not require visible flooding.
- river_or_coastal_overflow_context: true when the tile includes or is visibly
  connected to a river, stream, canal, drainage channel, reservoir, wetland,
  delta, estuary, coastline, or coastal lowland where overflow or drainage
  flooding is plausible.
- low_lying_or_poor_drainage_area: true when the tile appears coastal,
  floodplain-like, flat urban lowland, paddy-field dominated, wetland-like,
  drainage-channel dense, or poorly drained. Bali location context may support
  this judgment, but lower confidence if terrain is unclear.
- vegetation_or_soil_saturation: true when RGB/SWIR suggests wet soil,
  waterlogged cropland, saturated vegetation, muddy floodplain surfaces, or
  moisture-stressed/wet land. Do not set true merely because healthy vegetation
  is green.
- permanent_water_body_present: true for normal ocean/coastline, rivers, canals,
  reservoirs, ponds, wetlands, fishponds, lakes, or stable water bodies. This
  helps separate normal water from possible floodwater.
- cloud_shadow_or_image_quality_limited: true when cloud, cloud shadow, haze,
  bright glare, no-data, or sensor/illumination artifacts limit reliable
  interpretation of land/water/exposure features.
- confidence:
    - high: clear imagery and strong evidence for most field decisions.
    - medium: usable imagery but partial cloud, mixed land/water context, or
      some ambiguity.
    - low: cloud/haze/no-data dominates, important features are obscured, or
      flood vs permanent-water distinction is uncertain.

Consistency rules:
- Do not mark all exposure/context fields false just because active floodwater
  is absent.
- Dense urban Denpasar-style tiles should usually have
  urban_or_infrastructure_exposure=true.
- Coastal/ocean tiles should usually have standing_water_present=true,
  permanent_water_body_present=true, and river_or_coastal_overflow_context=true,
  but temporary_inundation_likely=false unless water appears abnormal on land.
- Cloud-heavy tiles should not receive confidence=high.
- If unsure between two risk levels, choose the lower risk and lower confidence.
"""


def empty_label() -> dict[str, object]:
    """Return a low-risk label template in canonical field order."""
    return {
        "flood_risk_level": "low",
        "water_extent_level": "small",
        "standing_water_present": False,
        "temporary_inundation_likely": False,
        "urban_or_infrastructure_exposure": False,
        "road_or_transport_disruption_likely": False,
        "cropland_or_settlement_exposure": False,
        "river_or_coastal_overflow_context": False,
        "low_lying_or_poor_drainage_area": False,
        "vegetation_or_soil_saturation": False,
        "permanent_water_body_present": False,
        "cloud_shadow_or_image_quality_limited": False,
        "confidence": "medium",
    }


def validate_label(data: dict[str, Any]) -> dict[str, object]:
    """Validate a flood label and return it in canonical field order."""
    if not isinstance(data, dict):
        raise ValueError("label must be a JSON object")

    keys = set(data)
    expected = set(LABEL_FIELDS)
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"extra fields: {', '.join(sorted(extra))}")

    if data["flood_risk_level"] not in RISK_LEVELS:
        raise ValueError("flood_risk_level must be one of: low, medium, high")
    if data["water_extent_level"] not in WATER_EXTENT_LEVELS:
        raise ValueError("water_extent_level must be one of: small, moderate, large")
    if data["confidence"] not in CONFIDENCE_LEVELS:
        raise ValueError("confidence must be one of: low, medium, high")

    for field in BOOLEAN_FIELDS:
        if type(data[field]) is not bool:
            raise ValueError(f"{field} must be true or false")

    return OrderedDict((field, data[field]) for field in LABEL_FIELDS)


def dumps_label(data: dict[str, Any], *, indent: int | None = None) -> str:
    """Serialize a validated label in canonical field order."""
    return json.dumps(validate_label(data), indent=indent)


def load_label_text(text: str) -> dict[str, object]:
    """Parse and validate label JSON text."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"annotation is invalid JSON: {exc}") from exc
    return validate_label(raw)
