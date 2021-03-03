import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.shape_base import _accumulate

caminho_imagem = 'rsc/mulher.png'

if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]

img = cv.imread(cv.samples.findFile(caminho_imagem))

imagem_original = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
histograma_original = cv.calcHist(imagem_original, [0], None, [255], (0, 255))

plt.subplot(3, 3, 1)
plt.plot(histograma_original)
plt.title('Histograma original')
cv.imshow('Imagem original', imagem_original)

imagem_filtro_sigma = cv.bilateralFilter(imagem_original, 1, 100, 100)
histograma_filtro_sigma = cv.calcHist(imagem_filtro_sigma, [0], None, [255], (0, 255))

plt.subplot(3, 3, 2)
plt.plot(histograma_filtro_sigma)
plt.title('Histograma filtro sigma')
cv.imshow('Imagem filtro sigma', imagem_filtro_sigma)

linhas, colunas = imagem_filtro_sigma.shape
imagem_equalizada = np.uint8(np.zeros((linhas, colunas)))
lista_r = [0, 0.1, 0.5, 1.0, 1.5, 2]

for i, r in enumerate(lista_r):

    histograma_com_potencia_calculada = [w ** r for w in histograma_filtro_sigma]

    q = np.sum(histograma_com_potencia_calculada)
    param = (255 / q)

    somatorio_para_cada_valor_no_histograma = []
    for valor in range(len(histograma_com_potencia_calculada)):
        somatorio_para_cada_valor_no_histograma.append(np.sum(histograma_com_potencia_calculada[0:valor]))

    for x in range(linhas):
        for y in range(colunas):
            u = imagem_filtro_sigma[x][y]
            imagem_equalizada[x][y] = param * somatorio_para_cada_valor_no_histograma[u]

    histograma_imagem_equalizada = cv.calcHist(imagem_equalizada, [0], None, [255], (0, 255))

    plt.subplot(3, 3, i+3)
    plt.plot(histograma_imagem_equalizada)
    plt.title(f'Equalizada r={r}')
    cv.imshow(f'Imagem equalizada {r}', imagem_equalizada)

k = None
while k != 27:
    k = cv.waitKey(1) & 0xFF

plt.show()

