"""Concatenate clips with title cards into one reel.
usage: python make_reel.py OUT.mp4 SPEC.json   (SPEC: {"title": [...lines], "clips": [{"video": path, "card": [...lines]}]})
"""
import json, sys
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1520, 720


def font(n):
    try:
        return ImageFont.load_default(size=n)
    except TypeError:
        return ImageFont.load_default()


def card(lines, n=60):
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    y = H // 2 - 26 * len(lines)
    for i, t in enumerate(lines):
        d.text((50, y + 52 * i), t, fill=(31, 31, 30), font=font(30 if i == 0 else 22))
    return [np.asarray(im)] * n


def fit(fr):
    im = Image.fromarray(fr)
    if im.size != (W, H):
        s = min(W / im.width, H / im.height)
        im2 = Image.new("RGB", (W, H), "white")
        im = im.resize((int(im.width * s), int(im.height * s)))
        im2.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        im = im2
    return np.asarray(im)


def main(out, spec):
    S = json.load(open(spec))
    w = imageio.get_writer(out, fps=30, macro_block_size=1)
    for fr in card(S["title"], n=90):
        w.append_data(fr)
    for c in S["clips"]:
        for fr in card(c["card"], n=45):
            w.append_data(fr)
        for fr in imageio.get_reader(c["video"]):
            w.append_data(fit(fr))
    w.close()
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
