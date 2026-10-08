"""Tile per-method pour videos into one 2x2 video (BITSI v3 | whole object / FitPart+SAM 3 | SAM 3),
each tile with a header naming the method and its outcome; shorter clips hold their last frame.
usage: python compare_grid.py OUT.mp4 TITLE path:label:outcome x4
"""
import sys
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(n):
    try:
        return ImageFont.load_default(size=n)
    except TypeError:
        return ImageFont.load_default()


def main():
    out, title, specs = sys.argv[1], sys.argv[2], [a.split(":", 2) for a in sys.argv[3:7]]
    clips = [imageio.mimread(p, memtest=False) for p, _, _ in specs]
    T = max(len(c) for c in clips)
    tw, th, hh = 640, 480, 44
    frames = []
    for f in range(T):
        canvas = Image.new("RGB", (2 * tw, 2 * (th + hh) + 56), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        d.text((16, 14), title, fill=(31, 31, 30), font=font(26))
        for k, ((_, label, outcome), clip) in enumerate(zip(specs, clips)):
            x0, y0 = (k % 2) * tw, 56 + (k // 2) * (th + hh)
            im = Image.fromarray(clip[min(f, len(clip) - 1)])
            sc_ = min(tw / im.width, th / im.height)                # keep the aspect ratio (letterbox)
            im = im.resize((int(im.width * sc_), int(im.height * sc_)))
            canvas.paste(im, (x0 + (tw - im.width) // 2, y0 + hh + (th - im.height) // 2))
            good = outcome.startswith("SUCCESS")
            info = label.startswith("What")  # informational tile, not an outcome
            d.rectangle([x0, y0, x0 + tw - 1, y0 + hh - 1],
                        fill=(110, 110, 105) if info else (27, 150, 90) if good else (196, 0, 58))
            d.text((x0 + 12, y0 + 9), f"{label}: {outcome}", fill=(255, 255, 255), font=font(21))
        frames.append(np.asarray(canvas))
    imageio.mimsave(out, frames, fps=30, macro_block_size=1)
    print("saved", out, T, "frames")


if __name__ == "__main__":
    main()
