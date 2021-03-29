# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:46:40 2020

@author: baam4
"""

import pandas as pd 
import numpy as np
#import math
import matplotlib.pyplot as plt
from numpy.linalg import eig

             ##### Lecture du fichier
                    

df =pd.read_csv('filedonnees_2D.csv', sep=';') 
donn = df.to_numpy()



                ##### Données partie C

Xmoy = np.mean(donn, axis = 1) 
X = (donn.T - Xmoy).T


N = X.shape[1]
X2 = np.sqrt((np.sum(np.square(X),1))/N)
print(X2)
donn3 = (X.T / X2).T

print(df)
abscisse = donn3[0,:]
ordonnee = donn3[1,:]
plt.scatter(abscisse, ordonnee)
plt.title("Affichage des données en 2D")

            ####### Reponse aux questions



            ##### calcul de la matrice de correlation

CorrMatrix = np.corrcoef(donn3)
print(CorrMatrix)


            ##### calcul des valeurs et vecteurs propres

[a,b] = np.linalg.eig(CorrMatrix)


valordre = np.argsort (-a) # fournit les indices des valeurs triées par ordre croissant
b = b[ :,valordre] # ordonne vect_p selon les indices fournis par valordre
a = a [valordre]


print("valeur propres", a)
print("vecteurs propres", b)


            ##### projection

projection = np.matmul(b.T, donn3)
print(projection)

plt.show()
plt.scatter(projection[0,:], projection[1,:])
plt.title("Affichage des projections partie C")

variables = ['D1', 'D2','D3','D4','D5','D6','D7','D8', 'D9', 'D10', 'D11', 'D12']

for i in range(12):

    plt.text(projection[0,i],projection[1,i],variables[i],va="bottom",ha="center",fontsize=8)

r1 = np.zeros(12)
r2 = np.zeros(12)

for i in range(2):
    r1[i] = np.sqrt(a[0])*b[i,0]
    r2[i] = np.sqrt(a[1])*b[i,1]
    
    
print("affichage R1", r1)
print("affichage R2", r2)

    
            ##### Cercle de correlation
            

plt.figure()
theta = np.linspace(0, 2*np.pi, 100)
x1 = np.cos(theta)
x2 = np.sin(theta)
fig, ax = plt.subplots(1)
ax.plot(x1, x2)
ax.set_aspect(1)
plt.xlim(-1.05,1.05)
plt.ylim(-1.05,1.05)
plt.plot(r1,r2,'*')
for i in range(len(variables)) :
 plt.text(r1[i],r2[i],variables[i],va="bottom",ha="center",fontsize=8)
plt.grid(linestyle='-')
plt.title('cercle des corrélations partie C', fontsize=8)
plt.show()