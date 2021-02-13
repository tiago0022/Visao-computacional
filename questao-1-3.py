import sys

import cv2 as cv
import numpy as np

caminho_imagem = 'starry_night.png'
if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]
img = cv.imread(cv.samples.findFile(caminho_imagem))

cv.namedWindow('image')
cv.imshow('image', img.copy())


def evento_mouse(event, x, y, flags, param):

    tamanho = 11
    termo = int(tamanho/2)
    x0, x1, y0, y1 = map(lambda t: max(0, t), [x - termo, x + termo, y - termo, y + termo])
    conjunto_pixel = img[y0:y1, x0:x1]

    b, g, r = img[y, x]
    intensidade = (int(r) + int(g) + int(b)) / 3

    media = np.average(conjunto_pixel)
    desvio_padrao = np.std(conjunto_pixel)

    print(f'Ponto ({x}, {y})')
    print('RGB:', [r, g, b])
    print('Intensidade:', round(intensidade, 2))
    print('Média:', round(media, 2))
    print('Desvio padrão:', round(desvio_padrao, 2))
    print()

    copia = img.copy()
    cv.rectangle(copia, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (0, 255, 0))
    cv.imshow('image', copia)


k = None
while k != 27:
    cv.setMouseCallback('image', evento_mouse)
    k = cv.waitKey(1) & 0xFF
