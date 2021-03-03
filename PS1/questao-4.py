from sympy import Plane, Point3D, symbols
from sympy.plotting import plot3d
import numpy as np
import cv2 as cv
import numpy as np

# Não terminei esta questão, apenas gerei os planos

for u in range(255):
    plano = np.zeros((255, 255, 3), np.uint8)
    for vermelho in range(255):
        for verde in range(255):
            azul = u - vermelho - verde
            if azul > 0:
                plano[vermelho][verde][0] = vermelho
                plano[vermelho][verde][1] = verde
                plano[vermelho][verde][2] = azul

# cv.imshow("Display window", plano)
# k = None
# while k != 27:
#     k = cv.waitKey(1) & 0xFF
