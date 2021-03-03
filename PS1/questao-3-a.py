import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

caminho_imagem_1 = 'rsc/agua.png'
caminho_imagem_2 = 'rsc/fogo.png'

if len(sys.argv) >= 2:
    caminho_imagem_1 = sys.argv[1]
if len(sys.argv) >= 3:
    caminho_imagem_2 = sys.argv[2]

img_1 = cv.imread(caminho_imagem_1, 0)
img_2 = cv.imread(caminho_imagem_2, 0)

f_1 = np.fft.fft2(img_1)
fshift_1 = np.fft.fftshift(f_1)
fase_1 = np.angle(fshift_1)
amplitude_1 = 20*np.log(np.abs(fshift_1))

f_2 = np.fft.fft2(img_2)
fshift_2 = np.fft.fftshift(f_2)
fase_2 = np.angle(fshift_2)
amplitude_2 = 20*np.log(np.abs(fshift_2))

misto_1 = np.multiply(amplitude_2, np.exp(1j*fase_1))
img_mista_1 = np.real(np.fft.ifft2(misto_1))
img_mista_1 = np.abs(img_mista_1)

misto_2 = np.multiply(amplitude_1, np.exp(1j*fase_2))
img_mista_2 = np.real(np.fft.ifft2(misto_2))
img_mista_2 = np.abs(img_mista_2)

plt.imshow(img_mista_1, cmap='gray')
plt.show()

plt.imshow(img_mista_2, cmap='gray')
plt.show()