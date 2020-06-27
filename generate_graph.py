import math
import cv2
import pywt
import csv
import random
import numpy as np
from Util import Util
import matplotlib.pyplot as plt

if __name__ == "__main__":
    util = Util()
    #File = "C:\Users\Fabricio\PycharmProjects\Waveletfaces Parametrizáveis\RESULTADOS\Overall Accuracy per Wavelet\YaleB.txt_1D_Parametrization_lvl_4_pca_False_lda_True.csv"
    File = "C:\\Users\\fabri\\Python Projects\\Waveletfaces Parametrizaveis\\RESULTADOS\\2 Dimensions\\1296 iteracoes\\CSVs\\AR.txt_2D_Parametrization_lvl_3_pca_False_lda_True.csv"
    PI = np.pi
    b = False

    x_ini, y_ini = 0, 0
    x_end, y_end = 2* PI, 2 * PI
    y_ini = 0
    y_end = 100

    x, y, rfc, knn, gnb, svc = [], [], [], [], [], []

    with open(File, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for line in reader:
            #if b == False:
            #    b = True
            #    continue
            x.append(line[0])
            y.append(line[1])
            rfc.append(line[2])
            knn.append(line[4])
            gnb.append(line[6])
            svc.append(line[8])
            
    #util.plot2D("YaleB", x, knn, svc, gnb, rfc, x_ini, x_end, y_ini, y_end)
    util.plot3D(File, x, y, rfc, knn, svc, gnb, plt.cm.gray)
