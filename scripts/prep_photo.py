import os, sys
import cv2
import numpy as np
from PIL import Image
from rembg import remove
HERE = os.path.dirname(os.path.abspath(__file__))
inp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "Adas.png")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "adas-prepped.png")
CROP = (0.10, 0.11, 0.80, 0.82)
lw = 1.2
im = Image.open(inp).convert("RGBA"); W, H = im.size
im = im.crop((int(W*CROP[0]), int(H*CROP[1]), int(W*CROP[2]), int(H*CROP[3])))
cut = remove(im)
rgb = np.array(cut.convert("RGB")); alpha = np.array(cut.split()[-1])
s = 1000 / rgb.shape[1]
rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]))
g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(g)
sm = cv2.medianBlur(g, 5)
for _ in range(4): sm = cv2.bilateralFilter(sm, 9, 30, 7)
lo, hi = np.percentile(sm[alpha > 128], [2, 97])
P = np.clip((sm.astype(np.float32) - lo) / (hi - lo), 0, 1) ** 0.8
fine = cv2.GaussianBlur(sm, (0, 0), 2).astype(np.float32)
coarse = cv2.GaussianBlur(sm, (0, 0), 7).astype(np.float32)
lines = np.clip((coarse - fine) / 20.0, 0, 1)
P = np.clip(P - lw * lines, 0, 1)
L = (1 - P) * 0.74
m = cv2.GaussianBlur(alpha.astype(np.float32) / 255, (0, 0), 1)
o = (L * m + 1.0 * (1 - m)) * 255
h, w = o.shape; S = max(h, w); c = np.full((S, S), 255, np.uint8)
c[(S-h)//2:(S-h)//2+h, (S-w)//2:(S-w)//2+w] = o.astype(np.uint8)
Image.fromarray(c, "L").save(out)
print("wrote", out, c.shape)