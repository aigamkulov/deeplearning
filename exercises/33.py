import numpy as np
import cv2

def dominant_freq(row):
    x = row - row.mean()
    F = np.fft.rfft(x)
    k = np.argmax(np.abs(F))
    return k / len(x), 2 * np.abs(F[k]) / len(x)

n = np.arange(512)
img = (127.5 + 100 * np.cos(2 * np.pi * 0.3125 * n))[None, :].repeat(512, 0)
f0, a0 = dominant_freq(img[0])
sl = img[::4, ::4]
ar = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
f1, a1 = dominant_freq(sl[0])
f2, a2 = dominant_freq(ar[0])
theory = abs(np.exp(2j * np.pi * 0.3125 * np.arange(4)).sum()) / 4 * 100
print(f"original: f={f0:.4f} A={a0:.2f};  slicing ::4: f={f1:.4f} A={a1:.2f};  "
      f"INTER_AREA: f={f2:.4f} A={a2:.2f} (theory {theory:.2f})")
t = np.arange(20)
print("cos(2pi*0.7n) == cos(2pi*0.3n) on samples:",
      bool(np.allclose(np.cos(2 * np.pi * 0.7 * t), np.cos(2 * np.pi * 0.3 * t))))
