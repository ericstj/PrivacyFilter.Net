"""Generates the PrivacyFilter.Net package icon (original artwork).

A rounded-square gradient badge containing a shield with text lines and a
highlighted sensitive span. Represents privacy-aware text filtering and
redaction. Intentionally distinct from any upstream logo.
"""
from pathlib import Path

from PIL import Image, ImageDraw

S = 512  # supersample, downscaled at the end
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Vertical gradient background: indigo -> teal
top = (79, 70, 229)      # #4F46E5
bot = (14, 165, 165)     # #0EA5A5
for y in range(S):
    t = y / (S - 1)
    r = round(top[0] + (bot[0] - top[0]) * t)
    g = round(top[1] + (bot[1] - top[1]) * t)
    b = round(top[2] + (bot[2] - top[2]) * t)
    draw.line([(0, y), (S, y)], fill=(r, g, b, 255))

# Rounded-square mask
radius = int(S * 0.22)
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=radius, fill=255)
img.putalpha(mask)

draw = ImageDraw.Draw(img)

white = (255, 255, 255, 255)
text = (79, 70, 229, 255)
accent = (245, 197, 66, 255)

# Shield - represents protection of sensitive text.
shield = [
    (int(S * 0.50), int(S * 0.14)),
    (int(S * 0.79), int(S * 0.25)),
    (int(S * 0.76), int(S * 0.57)),
    (int(S * 0.68), int(S * 0.72)),
    (int(S * 0.50), int(S * 0.86)),
    (int(S * 0.32), int(S * 0.72)),
    (int(S * 0.24), int(S * 0.57)),
    (int(S * 0.21), int(S * 0.25)),
]
draw.polygon(shield, fill=white)

# Text lines inside the shield. The accent bar represents detected/redacted PII.
lx = int(S * 0.33)
lh = int(S * 0.045)
line_radius = lh // 2
draw.rounded_rectangle(
    [lx, int(S * 0.34), int(S * 0.68), int(S * 0.34) + lh],
    radius=line_radius,
    fill=text,
)
draw.rounded_rectangle(
    [lx, int(S * 0.46), int(S * 0.71), int(S * 0.46) + int(S * 0.07)],
    radius=int(S * 0.025),
    fill=accent,
)
draw.rounded_rectangle(
    [lx, int(S * 0.61), int(S * 0.61), int(S * 0.61) + lh],
    radius=line_radius,
    fill=text,
)

out = img.resize((128, 128), Image.Resampling.LANCZOS)
out.save(Path(__file__).with_name("icon.png"), "PNG")
print("wrote eng/icon.png")
