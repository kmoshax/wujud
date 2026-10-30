"""The README banner, one ASCII canvas: the name sends out Wi-Fi from the dot of its j, the rings travel
across the room, and where they meet a walking person they bend around the body. Person and name
are drawn in the hex the board prints.

    cd research && .venv/bin/python tools/banner.py   # writes docs/banner.png
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "..", "docs", "banner.png")
FONTS = "/usr/share/fonts/TTF/"
TEXT_FONT = FONTS + "JetBrainsMonoNLNerdFontMono-Regular.ttf"
NAME_FONT = FONTS + "JetBrainsMonoNLNerdFontMono-ExtraBold.ttf"
COLS, ROWS = 150, 30
CELL_W, CELL_H = 11, 20
# light to dense; the dense end is the hex the board prints
RAMP = " .'`,:;-~=+*il1r7vcz4f2e6593a0b8d"
HEX = "0123456789abcdef"


def capsule(px, py, a, b, r):
    ax, ay = a
    vx, vy = b[0] - ax, b[1] - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(px - ax - t * vx, py - ay - t * vy) - r


def walking_person(px, py):
    """Signed distance of a person mid-stride, side view, facing right; unit = person height."""
    parts = [
        np.hypot(px - 0.03, py - 0.91) - 0.075,
        capsule(px, py, (0.0, 0.80), (0.015, 0.55), 0.095),
        capsule(px, py, (0.0, 0.53), (0.13, 0.30), 0.062),
        capsule(px, py, (0.13, 0.30), (0.17, 0.05), 0.05),
        capsule(px, py, (0.0, 0.53), (-0.08, 0.29), 0.062),
        capsule(px, py, (-0.08, 0.29), (-0.20, 0.08), 0.05),
        capsule(px, py, (0.17, 0.04), (0.24, 0.035), 0.022),
        capsule(px, py, (-0.21, 0.07), (-0.15, 0.03), 0.022),
        capsule(px, py, (0.0, 0.78), (0.10, 0.62), 0.045),
        capsule(px, py, (0.10, 0.62), (0.16, 0.48), 0.04),
        capsule(px, py, (0.0, 0.78), (-0.09, 0.63), 0.045),
        capsule(px, py, (-0.09, 0.63), (-0.13, 0.49), 0.04),
    ]
    return np.min(parts, axis=0)


def lit(sd, scale):
    """Distance inside a shape read as height: a rounded volume lit from the upper left."""
    inside = np.clip(-sd / scale, 0, 1)
    gy, gx = np.gradient(-sd)
    norm = np.hypot(gx, gy) + 1e-12
    light = np.clip((-gx / norm * 0.6 + gy / norm * 0.8) * (1 - inside) + inside, 0, 1)
    return np.where(sd < 0, 0.3 + 0.7 * light * (0.45 + 0.55 * inside), 0.0)


def name_mask(cols, rows):
    """Coverage of each cell by the name (its j without a dot), and where the dot would be."""
    c0, c1 = cols
    r0, r1 = rows
    w, h = (c1 - c0) * CELL_W, (r1 - r0) * CELL_H
    image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(image)
    size = 400
    font = ImageFont.truetype(NAME_FONT, size)
    while draw.textbbox((0, 0), "wuȷud", font=font)[2] > w * 0.98:
        size -= 4
        font = ImageFont.truetype(NAME_FONT, size)
    box = draw.textbbox((0, 0), "wuȷud", font=font)
    x, y = (w - (box[2] - box[0])) // 2 - box[0], h - (box[3] - box[1]) - box[1] - int(0.12 * h)
    draw.text((x, y), "wuȷud", fill=255, font=font)
    dotted = Image.new("L", (w, h), 0)
    ImageDraw.Draw(dotted).text((x, y), "wujud", fill=255, font=font)
    ys, xs = np.nonzero(np.asarray(dotted, float) - np.asarray(image, float) > 128)
    dot = (xs.mean() / CELL_W + c0, ys.mean() / CELL_H + r0)
    pixels = np.asarray(image, float) / 255
    cover = pixels.reshape(r1 - r0, CELL_H, c1 - c0, CELL_W).mean(axis=(1, 3))
    return cover, dot


def main():
    rng = np.random.default_rng(11)
    chars = np.full((ROWS, COLS), " ", dtype="<U1")
    shade = np.zeros((ROWS, COLS))

    # the person, left of centre, facing the name; 4 x 2 samples per cell
    cy, cx = np.mgrid[0:ROWS * 4, 0:COLS * 2]
    px = (cx / 2 - 30) * CELL_W / (ROWS * CELL_H)
    py = 1.0 - cy / (ROWS * 4) * 1.06
    sd = walking_person(px, py)
    body = lit(sd, 0.05).reshape(ROWS, 4, COLS, 2).mean(axis=(1, 3))
    body_sd = sd.reshape(ROWS, 4, COLS, 2).min(axis=(1, 3))

    name_cols, name_rows = (68, 148), (1, 29)
    cover, (sx, sy) = name_mask(name_cols, name_rows)

    # rings from the j's dot, bent where they cross the body
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    dx, dy = (xx - sx) * CELL_W, (yy - sy) * CELL_H
    r = np.hypot(dx, dy)
    near_body = np.exp(-np.clip(body_sd, 0, None) / 0.035)
    spacing = 95.0
    bent = r + 16 * near_body * np.sin(yy * 1.3)
    off_ring = np.abs(bent - spacing * np.round(bent / spacing))
    angle = np.arctan2(dy, -dx)
    letters = np.zeros((ROWS, COLS), bool)
    letters[name_rows[0]:name_rows[1], name_cols[0]:name_cols[1]] = cover > 0.05
    around_letters = letters.copy()
    for shift in (-1, 1):
        around_letters |= np.roll(letters, shift, axis=0) | np.roll(letters, shift * 2, axis=1)
    on_ring = (off_ring < CELL_W * 0.55) & (r > 60) & (np.abs(angle) < 1.3) & ~around_letters
    fade = np.clip(1.1 - r / 1100, 0.15, 1)
    # the body shadows the signal: rings that pass behind it come out dimmer
    body_rows = body > 0.05
    behind = np.zeros_like(body_rows)
    for y in range(ROWS):
        if body_rows[y].any():
            behind[y, : np.nonzero(body_rows[y])[0].min()] = True
    for y, x in zip(*np.nonzero(on_ring)):
        a = angle[y, x]
        chars[y, x] = "(" if abs(a) < 0.4 else "-" if abs(a) > 1.0 else "/" if a < 0 else "\\"
        shadow = 0.35 if behind[y, x] else 1.0
        shade[y, x] = (0.42 * fade[y, x] + 0.5 * near_body[y, x]) * shadow

    # body over the rings: hex where it is lit, lighter marks at its edges
    for y, x in zip(*np.nonzero(body > 0.05)):
        k = int(np.clip(body[y, x], 0, 0.999) * (len(RAMP) - 1))
        chars[y, x] = RAMP[max(1, min(len(RAMP) - 1, k + rng.integers(-2, 3)))]
        shade[y, x] = 0.95

    # the name, filled with hex; the eye sits where the j's dot would be
    for y, x in zip(*np.nonzero(cover > 0.3)):
        cell = (y + name_rows[0], x + name_cols[0])
        chars[cell] = rng.choice(list(HEX)) if cover[y, x] > 0.7 else rng.choice(list(":=+il1"))
        shade[cell] = 0.95
    ex, ey = int(round(sx)), int(round(sy))
    for i, ch in enumerate("(@)"):
        chars[ey, ex - 1 + i] = ch
        shade[ey, ex - 1 + i] = 1.0

    font = ImageFont.truetype(TEXT_FONT, 17)
    image = Image.new("RGB", (COLS * CELL_W + 40, ROWS * CELL_H + 30), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for y in range(ROWS):
        for x in range(COLS):
            if chars[y, x] != " ":
                g = int(40 + 210 * shade[y, x])
                draw.text((20 + x * CELL_W, 15 + y * CELL_H), chars[y, x], fill=(g, g, g), font=font)
    image.save(OUT)
    print(os.path.relpath(OUT))


if __name__ == "__main__":
    main()
