"""Image quality helpers for fetched Sentinel-2 samples."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops


@dataclass(frozen=True)
class ImageQuality:
    path: str
    width: int
    height: int
    blank_fraction: float

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class PairQuality:
    rgb: ImageQuality
    swir: ImageQuality
    joint_blank_fraction: float

    @property
    def max_blank_fraction(self) -> float:
        return max(self.rgb.blank_fraction, self.swir.blank_fraction)

    def is_bad(self, blank_threshold: float) -> bool:
        return self.joint_blank_fraction >= blank_threshold


def image_blank_fraction(path: Path, pixel_threshold: int = 3) -> ImageQuality:
    """Measure the fraction of near-black no-data pixels in an RGB image."""
    with Image.open(path) as raw:
        image = raw.convert("RGB")
        width, height = image.size
        r, g, b = image.split()
        max_channel = ImageChops.lighter(ImageChops.lighter(r, g), b)
        histogram = max_channel.histogram()
        blank_count = sum(histogram[: pixel_threshold + 1])
        total = width * height
    return ImageQuality(
        path=str(path),
        width=width,
        height=height,
        blank_fraction=blank_count / total if total else 1.0,
    )


def pair_quality(rgb_path: Path, swir_path: Path, pixel_threshold: int = 3) -> PairQuality:
    """Measure no-data blank fractions for an RGB/SWIR image pair.

    A dark ocean can be near-black in SWIR while still being valid imagery.
    True no-data gaps from tile footprints are near-black in both composites,
    so repair decisions should use joint_blank_fraction instead of the maximum
    single-image blank fraction.
    """
    joint_blank_fraction = pair_joint_blank_fraction(rgb_path, swir_path, pixel_threshold)
    return PairQuality(
        rgb=image_blank_fraction(rgb_path, pixel_threshold),
        swir=image_blank_fraction(swir_path, pixel_threshold),
        joint_blank_fraction=joint_blank_fraction,
    )


def pair_joint_blank_fraction(
    rgb_path: Path,
    swir_path: Path,
    pixel_threshold: int = 3,
) -> float:
    """Return fraction of pixels that are near-black in both RGB and SWIR."""
    with Image.open(rgb_path) as raw_rgb, Image.open(swir_path) as raw_swir:
        rgb = raw_rgb.convert("RGB")
        swir = raw_swir.convert("RGB")
        if rgb.size != swir.size:
            raise ValueError(f"Image sizes differ: {rgb_path}={rgb.size}, {swir_path}={swir.size}")

        rgb_mask = _blank_mask(rgb, pixel_threshold)
        swir_mask = _blank_mask(swir, pixel_threshold)
        joint_mask = ImageChops.multiply(rgb_mask, swir_mask)
        blank_count = joint_mask.histogram()[255]
        total = rgb.size[0] * rgb.size[1]
    return blank_count / total if total else 1.0


def _blank_mask(image: Image.Image, pixel_threshold: int) -> Image.Image:
    r, g, b = image.split()
    max_channel = ImageChops.lighter(ImageChops.lighter(r, g), b)
    return max_channel.point(lambda value: 255 if value <= pixel_threshold else 0)
