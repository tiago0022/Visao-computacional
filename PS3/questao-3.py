import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

caminho_imagem = 'rsc/objetos2.png'

if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]

img = cv.imread(caminho_imagem, 0)

T_cor = 127
T_limite_diametro = 100

_, imagem_binaria = cv.threshold(img, T_cor, 255, 0)

lista_contorno, _ = cv.findContours(imagem_binaria, 1, 2)

lista_contorno_selecionado = []
lista_diametro = []

x = len(img[0]) - 1
y = len(img) - 1

for contorno in lista_contorno:
    area = round(cv.contourArea(contorno), 2)
    diametro = round(np.sqrt(4*area/np.pi))
    if diametro >= T_limite_diametro:
        # Verifica se o contorno é a própria imagem inteira
        if len(contorno) == 4 and (contorno[0] == [[0, 0]]).all() and (contorno[1] == [[0, y]]).all() and (contorno[2] == [[x, y]]).all() and (contorno[3] == [[x, 0]]).all():
            break
        lista_diametro.append(diametro)
        lista_contorno_selecionado.append(contorno)

imagem_contornos = cv.cvtColor(imagem_binaria, cv.COLOR_GRAY2BGR)

cv.drawContours(imagem_contornos, lista_contorno_selecionado, -1, (0, 255, 0), 1)
cv.imshow('imagem', imagem_contornos)


def evento_mouse(event, xe, ye, flags, param):
    if(event == cv.EVENT_LBUTTONDOWN):

        contorno = None
        for i in range(len(lista_contorno_selecionado)):
            if cv.pointPolygonTest(lista_contorno_selecionado[i], (xe, ye), False) >= 0:
                contorno = lista_contorno_selecionado[i]
                break

        if contorno is not None:

            m = cv.moments(contorno)

            # Centróide
            centroide_x = int(m['m10']/m['m00'])
            centroide_y = int(m['m01']/m['m00'])

            # Eixo principal
            (xe, ye), (e1, e2), angulo = cv.fitEllipse(contorno)
            e = max(e1, e2)/2
            angulo = angulo - 90 if angulo > 90 else angulo + 90
            xt = int(xe + math.cos(math.radians(angulo))*e)
            yt = int(ye + math.sin(math.radians(angulo))*e)
            xb = int(xe + math.cos(math.radians(angulo+180))*e)
            yb = int(ye + math.sin(math.radians(angulo+180))*e)

            # Ecentricidade
            numerador = ((m['m20'] - m['m02']) ** 2) - 4 * (m['m11'] ** 2)
            ecentricidade = numerador / ((m['m20'] + m['m02']) ** 2)

            print(f'\nPonto ({x}, {y})')
            print(f'Centróide: ({centroide_x}, {centroide_y})')
            print(f'Ecentricidade: {round(ecentricidade, 2)}')

            cv.line(imagem_contornos, (xt, yt), (xb, yb), (0, 255, 255), 3)
            cv.circle(imagem_contornos, (centroide_x, centroide_y), 1, (255, 0, 0), 3)
            cv.imshow('imagem', imagem_contornos)


k = None
while k != 27:
    cv.setMouseCallback('imagem', evento_mouse)
    k = cv.waitKey(1) & 0xFF
