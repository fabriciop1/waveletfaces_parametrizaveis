# -*- coding: utf-8 -*-

import math
import cv2
import pywt
import csv
import random
import time
import timeit
import warnings
import numpy as np
from Util import Util
from Waveletfaces import Waveletfaces
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# height = number of rows        width = number of columns

def findCoefficients_six(Theta1, Theta2):
    c0 = (1 + math.cos(Theta1) - math.cos(Theta2) - math.cos(Theta1) * math.cos(Theta2))/4 + (math.sin(Theta1)
                               - math.cos(Theta2) * math.sin(Theta1) - math.sin(Theta2) + math.cos(Theta1) * math.sin(Theta2) - math.sin(Theta1) * math.sin(Theta2))/4
    c1 = (1 - math.cos(Theta1) + math.cos(Theta2) - math.cos(Theta1) * math.cos(Theta2))/4 + (math.sin(Theta1)
                               + math.cos(Theta2) * math.sin(Theta1) - math.sin(Theta2) - math.cos(Theta1) * math.sin(Theta2) - math.sin(Theta1) * math.sin(Theta2))/4
    c2 = (1 + math.cos(Theta1) * math.cos(Theta2) + math.cos(Theta2) * math.sin(Theta1) - math.cos(Theta1) * math.sin(Theta2))/2 + (math.sin(Theta1) * math.sin(Theta2))/2
    c3 = (1 + math.cos(Theta1) * math.cos(Theta2))/2 + (-math.cos(Theta2) * math.sin(Theta1) + math.cos(Theta1) * math.sin(Theta2) + math.sin(Theta1) * math.sin(Theta2))/2
    c4 = (1 - math.cos(Theta1) + math.cos(Theta2) - math.cos(Theta1) * math.cos(Theta2) - math.sin(Theta1))/4 + (-math.cos(Theta2) * math.sin(Theta1) + math.sin(Theta2)
                               + math.cos(Theta1) * math.sin(Theta2) - math.sin(Theta1) * math.sin(Theta2))/4
    c5 = (1 + math.cos(Theta1) - math.cos(Theta2) - math.cos(Theta1) * math.cos(Theta2) - math.sin(Theta1) + math.cos(Theta2) * math.sin(Theta1))/4 +(math.sin(Theta2)
                               - math.cos(Theta1) * math.sin(Theta2) - math.sin(Theta1) * math.sin(Theta2))/4

    return [c0,c1,c2,c3,c4,c5]

def findCoefficients_four(Theta1):
    c0 = (1 - math.cos(Theta1) + math.sin(Theta1))/2
    c1 = (1 + math.cos(Theta1) + math.sin(Theta1))/2
    c2 = (1 + math.cos(Theta1) - math.sin(Theta1))/2
    c3 = (1 - math.cos(Theta1) - math.sin(Theta1))/2

    return [c0,c1,c2,c3]

def optimal_wavelet_2d(x_ini, x_end, y_ini, y_end, increment, level, method_lda, pcaM):
    x = []
    #knn = []
    #svc = []
    #gnb = []
    #rfc = []
    theta1 = x_ini
    i = 0

    write_file = open("%s_halfTraining_lvl%s_pca_%s_lda_%s_1D.csv" % (File, level, pcaM, method_lda), "ab")
    writer = csv.writer(write_file, delimiter = ";")

    wave_file = open("%s_1D_Parametrization_lvl_%s_pca_%s_lda_%s.csv" % (File, level, pcaM, method_lda), "ab")
    writer2 = csv.writer(wave_file, delimiter = ";")
    
    while theta1 < x_end:
        print "theta1 = ", theta1
        writer.writerow(["Theta: %s" % theta1])
        
        wavelet = pywt.Wavelet(filter_bank=(findCoefficients_four(theta1), [0,0,0,0], [0,0,0,0], [0,0,0,0]))
        wavelet.orthogonal = True
        
        taxas_acerto_rfc, taxas_acerto_svc, taxas_acerto_gnb, taxas_acerto_knn = waveletfaces.method(File, wavelet, lvl=level, method_lda=method_lda, pca=pcaM)
        
        x.append(theta1)
        knn_mean, svc_mean, gnb_mean, rfc_mean = np.mean(np.array(taxas_acerto_knn)), np.mean(np.array(taxas_acerto_svc)), np.mean(np.array(taxas_acerto_gnb)), np.mean(np.array(taxas_acerto_rfc))
        knn_std,svc_std,gnb_std,rfc_std = np.std(np.array(taxas_acerto_knn)),np.std(np.array(taxas_acerto_svc)),np.std(np.array(taxas_acerto_gnb)),np.std(np.array(taxas_acerto_rfc))
        #knn.append(knn_mean)
        #svc.append(svc_mean)
        #gnb.append(gnb_mean)
        #rfc.append(rfc_mean)

        lista = ["%.3f" % elem for elem in taxas_acerto_rfc]
        lista.insert(0, "RFC")
        writer.writerow(lista)
        lista = ["%.3f" % elem for elem in taxas_acerto_knn]
        lista.insert(0, "KNN")
        writer.writerow(lista)
        lista = ["%.3f" % elem for elem in taxas_acerto_gnb]
        lista.insert(0, "GNB")
        writer.writerow(lista)
        lista = ["%.3f" % elem for elem in taxas_acerto_svc]
        lista.insert(0, "SVM")
        writer.writerow(lista)

        writer2.writerow(["%s" % theta1, "%.3f" % rfc_mean, "%.5f" % rfc_std, "%.3f" % knn_mean, "%.5f" % knn_std, "%.3f" % gnb_mean, "%.5f" % gnb_std, "%.3f" % svc_mean, "%.5f" % svc_std])
        
        theta1 += increment
        i += 1

    write_file.close()
    wave_file.close()
        
    print "Numero de iteracoes -> ", i
    #print "Melhor acuracia por classificador: \tKNN = ", max(knn), " \tGNB = ", max(gnb), " \tSVC = ", max(svc), " \tRFC = ", max(rfc)
    #lista = [max(knn), max(gnb), max(svc), max(rfc)]
    #index = lista.index(max(lista))
    #if index == 0: pos = knn.index(max(knn))
    #elif index == 1: pos = gnb.index(max(gnb))
    #elif index == 2: pos = svc.index(max(svc))
    #else: pos = rfc.index(max(rfc))

    #print "\t\t\tMelhor Valor de X para o classificador com max valor = ", x[pos]
    print "\nTime: ", (timeit.default_timer() - start_time) / 60.0, " mins"

    #util.plot2D(File, x, knn, svc, gnb, rfc, x_ini, x_end, y_ini, y_end)
    
def optimal_wavelet_3d(x_ini, x_end, y_ini, y_end, level, increment_x, increment_y, method_lda, pcaM):
    x, y = [], []
    knn, svc, gnb, rfc = [], [], [], []
    theta1, theta2 = x_ini, y_ini
    i, j = 0, 0

    write_file = open("%s_halfTraining_lvl%s_pca_%s_lda_%s_2D.csv" % (File, level, pcaM, method_lda), "ab")
    writer = csv.writer(write_file, delimiter = ";")

    wave_file = open("%s_2D_Parametrization_lvl_%s_pca_%s_lda_%s.csv" % (File, level, pcaM, method_lda), "ab")
    writer2 = csv.writer(wave_file, delimiter = ";")

    while theta1 < x_end:
        x.append(theta1)
        #knn.append([])
        #gnb.append([])
        #svc.append([])
        #rfc.append([])
        while theta2 < y_end:
            if j == 0:
                y.append(theta2)
                
            writer.writerow(["Theta1: %s" % theta1, "Theta2: %s" % theta2])
            
            print "theta1 =", theta1, "theta2 =", theta2
            wavelet = pywt.Wavelet(filter_bank=(findCoefficients_six(theta1, theta2), [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]))
            taxas_acerto_rfc, taxas_acerto_svc, taxas_acerto_gnb, taxas_acerto_knn = waveletfaces.method(File, wavelet, lvl=level, method_lda=method_lda, pca=pcaM)

            knn_mean,svc_mean,gnb_mean,rfc_mean = np.mean(np.array(taxas_acerto_knn)),np.mean(np.array(taxas_acerto_svc)),np.mean(np.array(taxas_acerto_gnb)),np.mean(np.array(taxas_acerto_rfc))
            knn_std,svc_std,gnb_std,rfc_std = np.std(np.array(taxas_acerto_knn)),np.std(np.array(taxas_acerto_svc)),np.std(np.array(taxas_acerto_gnb)),np.std(np.array(taxas_acerto_rfc))

            #knn[j].append(knn_mean)
            #svc[j].append(svc_mean)
            #gnb[j].append(gnb_mean)
            #rfc[j].append(rfc_mean)

            lista = ["%.3f" % elem for elem in taxas_acerto_rfc]
            lista.insert(0, "RFC")
            writer.writerow(lista)
            lista = ["%.3f" % elem for elem in taxas_acerto_knn]
            lista.insert(0, "KNN")
            writer.writerow(lista)
            lista = ["%.3f" % elem for elem in taxas_acerto_gnb]
            lista.insert(0, "GNB")
            writer.writerow(lista)
            lista = ["%.3f" % elem for elem in taxas_acerto_svc]
            lista.insert(0, "SVM")
            writer.writerow(lista)

            writer2.writerow(["%s"%theta1, "%s"%theta2, "%.3f"%rfc_mean, "%.5f"%rfc_std, "%.3f"%knn_mean, "%.5f"%knn_std, "%.3f"%gnb_mean, "%.5f"%gnb_std, "%.3f"%svc_mean, "%.5f"%svc_std])

            theta2 += increment_y
            i += 1
                
        theta1 += increment_x
        theta2 = y_ini
        j += 1 

    write_file.close()
    wave_file.close()
    
    print "Numero de iteracoes -> ", i
    print "\nTime: ", (timeit.default_timer() - start_time) / 60.0, " mins"
    print "End Time: ", time.ctime()
    
    #x, y = np.array(x), np.array(y)
    
    #util.plot3D(File, x, y, knn, svc, gnb, rfc, plt.cm.jet)
    #util.plot3D(File, x, y, knn, svc, gnb, rfc, plt.cm.gray)
    
if __name__ == "__main__":
    start_time = timeit.default_timer()
    print "Start Time: ", time.ctime()
    util = Util()
    waveletfaces = Waveletfaces()
    holdouts = 100
    files = ["ORL.txt", "YaleB.txt", "GTech.txt", "faces95.txt", "AR.txt"]  # bases de dados
    #methodPCA, methodLDA, levels = [False, True], [False, True], [5]
    PI = np.pi
    File = files[4]
    
    #################################### 2D PLOT ###########################################
    #n_iteracoes = 512
    
    #x_ini = 0.0
    #x_end = 2 * PI
    #y_ini = 0
    #y_end = 100

    #for i in levels:
    #    for j in methodPCA:
    #        for k in methodLDA:
    #            optimal_wavelet_2d(x_ini, x_end, y_ini, y_end, increment= x_end / n_iteracoes, level=i, pcaM=j, method_lda=k)
                
    ########################################################################################

    #################################### 3D PLOT ###########################################
    n_iteracoes = 529 
    
    x_ini =  0
    x_end =  2 * PI
    y_ini = 0
    y_end = 2 * PI

    optimal_wavelet_3d(x_ini, x_end, y_ini, y_end, increment_x = 2 * PI / math.sqrt(n_iteracoes),
                       increment_y = 2 * PI / math.sqrt(n_iteracoes), level=3, pcaM=False, method_lda=True)
    ########################################################################################
