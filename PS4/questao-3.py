import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

n_colunas = 2
n_linhas = 3

global indice
indice = 0


def exibe(imagem, nome=''):
    global indice
    indice = indice + 1
    plt.subplot(n_linhas, n_colunas, indice), plt.imshow(imagem, cmap='gray')
    plt.title(nome), plt.xticks([]), plt.yticks([])


caminho_pasta_videos = 'rsc/xadrez/'
if len(sys.argv) >= 2:
    caminho_video = sys.argv[1]

iteracao = 1
img = cv.imread(caminho_pasta_videos + str(iteracao) + '.png')
contem_quadro = not (img is None)

while contem_quadro:

    # Quadro original
    img_cinza = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    img_exibida = cv.cvtColor(img_cinza.copy(), cv.COLOR_GRAY2BGR)

    # Bordas
    img_bordas = cv.Canny(img_cinza, 50, 150)

    # Linhas
    img_linhas = img_exibida.copy()
    linhas = cv.HoughLinesP(img_bordas, 1, np.pi/180, 80, minLineLength=80, maxLineGap=10)
    for linha in linhas:
        for x1, y1, x2, y2 in linha:
            cv.line(img_linhas, (x1, y1), (x2, y2), (255, 0, 0), 1, cv.LINE_AA)

    # Cantos
    img_cantos = img_linhas.copy()
    cantos = cv.goodFeaturesToTrack(img_cinza, 100, 0.01, 10, None, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)
    if img_cantos is not None:
        for i in range(cantos.shape[0]):
            cv.circle(img_cantos, (int(cantos[i, 0, 0]), int(cantos[i, 0, 1])), 3, (0, 255, 0), cv.FILLED)

    # Subcantos
    img_subcantos = img_cantos.copy()
    subcantos = cv.cornerSubPix(img_cinza, cantos, (5, 5), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 500, 0.0001))
    if img_subcantos is not None:
        for i in range(subcantos.shape[0]):
            cv.circle(img_subcantos, (int(subcantos[i, 0, 0]), int(subcantos[i, 0, 1])), 3, (0, 0, 255), cv.FILLED)

    exibe(img_subcantos, f'Quadro {iteracao}')

    iteracao += 1
    img = cv.imread(caminho_pasta_videos + str(iteracao) + '.png')
    contem_quadro = not (img is None)

plt.show()
