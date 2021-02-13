import sys

import cv2 as cv

caminho_imagem = 'rsc/starry_night.png'

if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]

img = cv.imread(cv.samples.findFile(caminho_imagem))

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)
cv.waitKey(0)
