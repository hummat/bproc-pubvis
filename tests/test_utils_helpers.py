from collections.abc import Sequence
from typing import cast

import numpy as np
from PIL import Image

from bproc_pubvis.constants import Color
from bproc_pubvis.utils import (
    _subsample_indices,
    _target_count,
    depth_to_image,
    get_color,
    normalize,
    set_background_color,
)


def test_target_count_fraction_and_bounds():
    assert _target_count(10, 0.5) == 5  # ceil(n * fraction)
    assert _target_count(3, 10) == 3  # clipped to n
    assert _target_count(5, 0) == 1  # clipped to minimum 1


def test_subsample_indices_random_and_fps():
    rng = np.random.default_rng(0)
    points = rng.standard_normal((10, 3))

    # Random method returns requested size, unique indices
    rand_idx = _subsample_indices(points, 0.4, method="random")
    assert len(rand_idx) == 4
    assert len(set(rand_idx)) == 4
    assert rand_idx.max() < len(points)

    # FPS method selects distinct points and respects target count
    rng = np.random.default_rng(1)
    np.random.seed(1)  # fps uses np.random for first pick
    fps_idx = _subsample_indices(points, 5, method="fps")
    assert len(fps_idx) == 5
    assert len(set(fps_idx)) == 5
    assert fps_idx.max() < len(points)


def test_normalize_and_depth_to_image():
    arr = np.array([2.0, 4.0, 6.0])
    normalized = normalize(arr)
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)

    depth = np.array([[0.0, 1.0], [2.0, 0.5]])
    img = depth_to_image(depth.copy(), cmap_name="Greys_r")
    assert isinstance(img, Image.Image)
    pixels = np.array(img)
    # Zero depth stays zero in all channels
    assert (pixels[0, 0] == 0).all()
    # Non-zero depths are non-zero grayscale values
    assert pixels[0, 1].max() > 0


def test_get_color_and_background_composite():
    # Enum and string resolution
    assert get_color(Color.WHITE) == (1.0, 1.0, 1.0)
    assert get_color("WHITE") == (1.0, 1.0, 1.0)

    rand = get_color("random")
    assert np.shape(rand) == (3,)
    rand_arr = np.asarray(rand)
    assert ((0 <= rand_arr) & (rand_arr <= 1)).all()

    rand_enum = get_color("random_color")
    assert rand_enum in [c.value for c in Color]

    # Background composite keeps opaque pixels, fills transparent ones
    base = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    base.putpixel((1, 0), (0, 0, 0, 255))
    result = set_background_color(base, Color.RED)
    pixels_raw = cast(Sequence[tuple[int, int, int, int]], result.convert("RGBA").getdata())
    res_pixels = list(pixels_raw)
    assert res_pixels[0] == (255, 0, 0, 255)  # filled from background color
    assert res_pixels[1] == (0, 0, 0, 255)  # original opaque pixel preserved
