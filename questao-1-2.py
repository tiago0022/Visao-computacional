import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

caminho_imagem = 'rsc/starry_night.png'

if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]

img = cv.imread(cv.samples.findFile(caminho_imagem))

bgr_planes = cv.split(img)
hist_size = 256

b_hist = cv.calcHist(bgr_planes, [0], None, [hist_size], (0, hist_size))
g_hist = cv.calcHist(bgr_planes, [1], None, [hist_size], (0, hist_size))
r_hist = cv.calcHist(bgr_planes, [2], None, [hist_size], (0, hist_size))

plt.plot(b_hist, 'b')
plt.plot(g_hist, 'g')
plt.plot(r_hist, 'r')

plt.show()
