import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import eig

df =pd.read_csv('temp_et_coord_14villes.csv', sep=';') 
donn = df.to_numpy()
print(donn)
temperatures = donn[0:12,:]
coordonnees = donn[12:14,:]
plt.plot(coordonnees[0,:],coordonnees[1,:],'.', markersize=10)
nomvilles = df.columns
print(nomvilles)

            ##### affichage des villes en fonction des coordonnées

for i in range(len(nomvilles)):

    plt.text(coordonnees[0,i],coordonnees[1,i],nomvilles[i],va="bottom",ha="center",fontsize=8)
 
plt.title("Figure 1") 

            ####   affichage des temperature par ville en fonction du mois          

plt.figure()
plt.plot(temperatures)
plt.title("Figure 2")

            #### calcul des valeurs centrees reduite

Xmoy = np.mean(temperatures, axis = 1) 
X = (temperatures.T - Xmoy).T
plt.figure()
plt.plot(X)

N = X.shape[1]
X2 = np.sqrt((np.sum(np.square(X),1))/N)
print(X2)
X3 = (X.T / X2).T
plt.figure()
plt.plot(X3)
plt.title("Figure 3")


                ##### Matrice de correlation

print(X3)

CorrMatrix3 = np.corrcoef(X3)
print(CorrMatrix3)


                ##### calcul et affichage valeur et vecteur propre

[a,b] = np.linalg.eig(CorrMatrix3)

print(a)
print(b)

                ##### Tracer la courbe de pourcentage d’inertie
               
sum_val_propre = 0
for i in range(12) :
    sum_val_propre = sum_val_propre + a[i]
    

print(sum_val_propre)

ratio = np.array([0,0,0,0,0,0,0,0,0,0,0,0], float)            
for i in range(12): 
    ratio[i] = (a[i])/sum_val_propre
    print(ratio[i])
        
plt.figure()
plt.plot(ratio)
plt.title("Figure 4")


 ###### Calcul de la projection des points sur les axes obtenus

projection = np.matmul(b.T, X3)
print(projection)

plt.show()
plt.scatter(projection[0,:], projection[1,:])

for i in range(len(nomvilles)):

    plt.text(projection[0,i],projection[1,i],nomvilles[i],va="bottom",ha="center",fontsize=8)

plt.title("Figure 5")

            ###### coefficient de correlation des 2 premieres composantes

r1 = np.zeros(len(ratio))
r2 = np.zeros(len(ratio))

for i in range(12):
    r1[i] = np.sqrt(a[0])*b[i,0]
    r2[i] = np.sqrt(a[1])*b[i,1]
    
            ##### Cercle de correlation
            
moisname = ['J', 'F','M','A','Mai','J','Jt','A', 'S', 'O', 'N', 'D']
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
for i in range(len(moisname)) :
 plt.text(r1[i],r2[i],moisname[i],va="bottom",ha="center",fontsize=8)
plt.grid(linestyle='-')
plt.title('cercle des corrélations', fontsize=8)
plt.show()