import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

caminho_imagem = 'rsc/objetos2.png'

if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]

img = cv.imread(caminho_imagem, 0)

T = 127

_, imagem_binaria = cv.threshold(img, T, 255, 0)

lista_contorno, _ = cv.findContours(imagem_binaria, 1, 2)

for i in range(len(lista_contorno)):
    lista_contorno[i] = np.uint64(lista_contorno[i])

contador_preto = 0
contador_branco = 0

for linha in imagem_binaria:
    for i in range(len(linha)):
        if linha[i] == 0:
            contador_preto += 1
        else:
            contador_branco += 1

plt.subplot(2, 1, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 1, 2), plt.imshow(imagem_binaria, cmap='gray')
plt.title(f'Binária - pretos: {contador_preto} / brancos: {contador_branco}'), plt.xticks([]), plt.yticks([])
plt.show()

cv.imshow('image', imagem_binaria)


def evento_mouse(event, x, y, flags, param):
    if(event == cv.EVENT_LBUTTONDOWN):

        print(f'Ponto ({x}, {y})')
        pixel = imagem_binaria[y][x]

        if pixel != 0:
            print('Pixel branco')
        else:
            contorno = None
            for i in range(len(lista_contorno)):
                if cv.pointPolygonTest(lista_contorno[i], (x, y), False) >= 0:
                    contorno = lista_contorno[i]
                    break

            try:
                area = round(cv.contourArea(contorno), 2)
                perimetro = round(cv.arcLength(contorno, True), 2)
                diametro = np.sqrt(4*area/np.pi)
            except:
                area = '<erro objeto muito grande>'
                perimetro = '<erro objeto muito grande>'
                diametro = '<erro objeto muito grande>'

            print('Área:', area)
            print('Perímetro:', perimetro)
            print('Diâmetro:', diametro)

        print()


k = None
while k != 27:
    cv.setMouseCallback('image', evento_mouse)
    k = cv.waitKey(1) & 0xFF
