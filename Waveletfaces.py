# -*- coding: utf8 -*-

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
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Waveletfaces:
    
    def method(self, File, wavelet, lvl, method_lda, pca):
        util = Util()
        taxas_acerto_svc = []
        taxas_acerto_knn = []
        taxas_acerto_gnb = []
        taxas_acerto_rfc = []

        with open(File, "rb") as csvfile:
            reader = csv.reader(csvfile, delimiter='\n')
            for line in reader:  # each holdout
                nn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
                gnb = GaussianNB()  
                svc = LinearSVC()
                rfc = RandomForestClassifier()

                treino, teste = [], []
                classes_treino, classes_teste = [], []

                lista = line[0].split(";")
                l = lista[0].split("|")
                l.remove("")

                for i in l:
                    treino.append(i.split(","))

                l = lista[1].split("|")
                l.remove("")

                for i in l:
                    teste.append(i.split(","))

                training = np.array(treino)
                test = np.array(teste)

                training_imgs = []
                test_imgs = []

                for row in training:
                    for f in row[1:]:
                        img = cv2.imread(f)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        coeffs = pywt.wavedec2(img, wavelet, level=lvl)
                        training_imgs.append(coeffs[0].flatten())
                        classes_treino.append(row[0])

                for row in test:
                    for f in row[1:]:
                        img = cv2.imread(f)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        coeffs = pywt.wavedec2(img, wavelet, level=lvl)
                        test_imgs.append(coeffs[0].flatten())
                        classes_teste.append(row[0])

                training_imgs = np.array(training_imgs)
                test_imgs = np.array(test_imgs)
                classes_treino = np.array(classes_treino)
                classes_teste = np.array(classes_teste)

                print "WAVELETFACES ", training_imgs.shape
                
                if pca:
                    if len(training_imgs[0]) < 50:
                        pca_m = PCA(n_components=len(training_imgs[0]))     # Método PCA   PCA reduz para 50 dimensões, caso o resultado do waveletfaces seja dimensão menor que 50, é mantido o valor
                    else:
                        pca_m = PCA(n_components=50)
                    pca_m.fit(training_imgs)
                    training_imgs = pca_m.transform(training_imgs)
                    test_imgs = pca_m.transform(test_imgs)
                    print "AFTER PCA ", training_imgs.shape
                    
                if method_lda:
                    lda = LinearDiscriminantAnalysis()  # Método LDA        Reduz para n_classes - 1 dimensões ou deixa como está caso seja menor
                    try:
                        lda.fit(training_imgs, classes_treino)
                        training_imgs = lda.transform(training_imgs)
                        test_imgs = lda.transform(test_imgs)
                        print "AFTER LDA", training_imgs.shape
                    except np.linalg.LinAlgError:
                        print "\nLinAlgError: SVD did not converge. Skipping.\n"
                        continue
                    
                print "SVC"
                svc.fit(training_imgs, classes_treino)
                preds = svc.predict(test_imgs)
                svc = None
                taxas_acerto_svc.append(util.getRealAccuracy(classes_teste, preds.astype(str)))
                print "GNB"
                gnb.fit(training_imgs, classes_treino)
                preds = gnb.predict(test_imgs)
                gnb = None
                taxas_acerto_gnb.append(util.getRealAccuracy(classes_teste, preds.astype(str)))
                print "NN"
                nn.fit(training_imgs, classes_treino)
                preds = nn.predict(test_imgs)
                nn = None
                taxas_acerto_knn.append(util.getRealAccuracy(classes_teste, preds.astype(str)))
                print "RFC"
                rfc.fit(training_imgs, classes_treino)
                preds = rfc.predict(test_imgs)
                rfc=None
                taxas_acerto_rfc.append(util.getRealAccuracy(classes_teste, preds.astype(str)))

        return taxas_acerto_rfc, taxas_acerto_svc, taxas_acerto_gnb, taxas_acerto_knn
