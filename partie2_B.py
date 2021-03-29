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


            ##### Données partie B 

donn2 =[donn[0,:]/100,donn[1,:]]
donn2 = np.array(donn2)




print(df)
abscisse = donn2[0,:]
ordonnee = donn2[1,:]
plt.scatter(abscisse, ordonnee)
plt.title("Affichage des données en 2D")


            ##### calcul de la matrice de correlation

CorrMatrix = np.corrcoef(donn2)
print(CorrMatrix)


            ##### calcul des valeurs et vecteurs propres

[a,b] = np.linalg.eig(CorrMatrix)


valordre = np.argsort (-a) # fournit les indices des valeurs triées par ordre croissant
b = b[ :,valordre] # ordonne vect_p selon les indices fournis par valordre
a = a [valordre]


print("valeur propres", a)
print("vecteurs propres", b)


            ##### projection

projection = np.matmul(b.T, donn2)
print(projection)

plt.show()
plt.scatter(projection[0,:], projection[1,:])
plt.title("Projection des données partie B")


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
plt.title('cercle des corrélations partie B', fontsize=8)
plt.show()