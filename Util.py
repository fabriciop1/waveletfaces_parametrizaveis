# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import csv
import os

class Util:

    def getAccuracy(self, testSet, predictions):     # APENAS 1-NN
        self.acertos = 0
        for i in range (len(testSet)):
            if (testSet[i, -1] == predictions[i]):
                self.acertos = self.acertos + 1
        return (self.acertos / float(len(testSet))) * 100.0

    def getRealAccuracy(self, realClasses, predictions):
        self.acertos = 0
        for i in range(len(realClasses)):
            if realClasses[i] == predictions[i]:
                self.acertos += 1
        return (self.acertos / float(len(realClasses))) * 100.0
    
    def write_csv(self, f, folder):  
        c = csv.writer(open(f, "wb"))
    
        SEPARATOR = ";"
        BASE_PATH = folder

        label = 0
        for dirname, dirnames, filenames in os.walk(BASE_PATH):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    abs_path = "%s/%s" % (subject_path, filename)
                    c.writerow(["%s%s%d" % (abs_path, SEPARATOR, label)])
                label = label + 1

    def write_results(self, filename, results):
        with open("results - 100 holdouts//%s.txt" %(filename), "w") as f:
            for i in results:
                f.write("%s\n" %i)
        
    def plot2D(self, File, x, knn, svc, gnb, rfc, x_ini, x_end, y_ini, y_end):
        fig, ax = plt.subplots()
        ax.plot(x, knn, 'bo-', label="1-NN")
        ax.plot(x, svc, 'rs-', label="Support Vector Classification")
        ax.plot(x, gnb, 'g^-', label="Gaussian Naive Bayes")
        ax.plot(x, rfc, 'kx-', label="Random Forest Classifier")
        legend = ax.legend(loc=4, shadow=True)
        plt.xlabel("Theta (rad)")
        plt.title("Base: %s" % File)
        plt.ylabel("Accuracy Rate (%)")
        plt.xlim(x_ini, x_end)
        plt.ylim(y_ini, y_end)
        plt.grid()
        plt.show()

    def plot3D(self, File, x, y, rfc, knn, svc, gnb, scale):
        plt.figure()

        plt.subplot(1,4,1)
        z = np.array(rfc).astype(float) / 100.0
        normi = mpl.colors.Normalize(vmin=np.amin(z), vmax=np.amax(z))
        plt.title("%s - RFC Accuracy" % File)
        plt.xlabel("Theta1")
        plt.ylabel("Theta2")
        plt.contourf(x,y,z.T, np.linspace(np.amin(z), np.amax(z), 300), cmap=scale, norm=normi, extend="both")
        plt.grid()
        plt.colorbar(format="%.2f")
        
        plt.subplot(1,4,2)
        z = np.array(knn) / 100.0
        normi = mpl.colors.Normalize(vmin=np.amin(z), vmax=np.amax(z))
        plt.title("%s - KNN Accuracy" % File)
        plt.xlabel("Theta1")
        plt.ylabel("Theta2")
        plt.contourf(x,y,z.T, np.linspace(np.amin(z), np.amax(z), 300), cmap=scale, norm=normi, extend="both")
        plt.grid()
        plt.colorbar(format="%.2f")
    
        plt.subplot(1,4,3)
        z = np.array(gnb) / 100.0
        normi = mpl.colors.Normalize(vmin=np.amin(z), vmax=np.amax(z))
        plt.title("%s - GNB Accuracy" % File)
        plt.xlabel("Theta1")
        plt.contourf(x,y,z.T, np.linspace(np.amin(z), np.amax(z), 300), cmap=scale, norm=normi, extend="both")
        plt.grid()
        plt.colorbar(format="%.2f")

        plt.subplot(1,4,4)
        z = np.array(svc) / 100.0
        normi = mpl.colors.Normalize(vmin=np.amin(z), vmax=np.amax(z))
        plt.title("%s - SVC Accuracy" % File)
        plt.contourf(x,y,z.T, np.linspace(np.amin(z), np.amax(z), 300), cmap=scale, norm=normi, extend="both")
        plt.xlabel("Theta1")
        plt.grid()
        plt.colorbar(format="%.2f")

        plt.show()
