"""Compose a same-physical-scale row of axon cross-sections.

The source PNGs (img1-4) are matplotlib cross-section figures, each covering a
different physical field of view (L x L um) but rendered at the same panel size.
Shown as-is they all look the same size, hiding the axon-diameter differences.

This script crops each image to a common physical window (WINDOW_UM) so that
1 um maps to the same number of pixels in every panel. Larger axons then look
larger, matching the layout of example_multi_axonsize.png.

Pipeline per image:
  1. Detect the square plot region (the axes-spine rectangle) in pixels.
     That square spans exactly L um.
  2. Centre-crop the central WINDOW_UM / L fraction -> a WINDOW_UM x WINDOW_UM view.
  3. Resize every crop to a common pixel size for uniform panels.
Then lay the crops out in a single row with an AxD label under each.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# array500_fibers_boxL_74.0_2026-05-09_20-52-56_avf_0.65_d1.5_sig0.4017wo10_wc3_wl3_K200_ODI_0.0032
# array500_fibers_boxL_140.0_2026-05-11_03-44-14_avf_0.67_d2.5_sig0.6695wo10_wc3_wl3_K200_ODI_0.0032_8871.58_sec.png
# array500_fibers_boxL_223.0_2026-06-02_06-03-15_avf_0.66_d3.5_sig0.9375wo10_wc3_wl3_K200_ODI_0.0032.png
# array500_fibers_boxL_311.0_2026-05-27_15-42-19_avf_0.68_d4.5_sig1.205wo10_wc3_wl3_K200_ODI_0.0032.png

# --- inputs -----------------------------------------------------------------
FILES = ["img1.png", "img2.png", "img3.png", "img4.png"]
FIELD_SIZES = [74.0, 140.0, 223.0, 311.0]      # physical L (um) of each image
AXON_DIAMETERS = [1.68, 2.58, 3.5, 4.5]        # mean axon diameter (um) per image

WINDOW_UM = min(FIELD_SIZES)                    # common physical window (74 um)
PANEL_PX = 1000                                 # output pixels per panel side
OUT_FILE = "multi_axonsize.png"


def _longest_run(mask_1d):
    """Length of the longest contiguous True run in a 1-D boolean array."""
    best = cur = 0
    for v in mask_1d:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def detect_plot_region(gray, frac=0.70, dark=200):
    """Return (r0, r1, c0, c1) bounding box of the square axes/plot region.

    The matplotlib axes spine is a near-full-width/height dark rectangle, much
    longer than any run produced by the circles or the tick/title text. Rows and
    columns whose longest dark run exceeds `frac` of the image span belong to the
    top/bottom and left/right spines, so their min/max give the plot bbox.
    """
    H, W = gray.shape
    dmask = gray < dark
    row_run = np.array([_longest_run(dmask[r]) for r in range(H)])
    col_run = np.array([_longest_run(dmask[:, c]) for c in range(W)])
    rows = np.where(row_run >= frac * W)[0]
    cols = np.where(col_run >= frac * H)[0]
    if rows.size == 0 or cols.size == 0:
        raise RuntimeError("Could not detect the plot frame; adjust frac/dark.")
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def crop_to_window(path, field_um, window_um, out_px):
    """Crop `path` to a central `window_um` view and resize to out_px square."""
    rgb = Image.open(path).convert("RGB")
    gray = np.asarray(rgb.convert("L"))
    r0, r1, c0, c1 = detect_plot_region(gray)

    plot = rgb.crop((c0, r0, c1, r1))          # square region spanning field_um
    w, h = plot.size
    frac = window_um / field_um                # fraction of the field to keep
    cw, ch = int(round(w * frac)), int(round(h * frac))
    left = (w - cw) // 2
    top = (h - ch) // 2
    view = plot.crop((left, top, left + cw, top + ch))
    return view.resize((out_px, out_px), Image.LANCZOS)


def main():
    crops = [
        crop_to_window(f, L, WINDOW_UM, PANEL_PX)
        for f, L in zip(FILES, FIELD_SIZES)
    ]

    n = len(crops)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.4))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.12, wspace=0.05)

    for ax, crop, d in zip(axes, crops, AXON_DIAMETERS):
        ax.imshow(np.asarray(crop))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(f"{d:g} \u00b5m", fontsize=24)

    fig.savefig(OUT_FILE, dpi=300)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()