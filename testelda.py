# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:14:33 2016

@author: josi
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.lda import LDA
import matplotlib.pyplot as plt

#GRUPOS ESCOLHIDOS
#g = [90, 31, 83, 50, 37, 89, 57, 14, 55, 16]
#g = [90, 31, 83, 50, 37, 89, 57, 14, 55, 16, 96, 87, 69, 98, 91]
#GRUPOS SELECIONADO PELO HYBRID
g = [34, 23, 17, 91, 99, 87, 52, 51, 19, 5]
#, 2, 100, 66, 45, 32]
#, 22, 82, 80, 76, 71, 69, 61, 54, 37, 29]


#g = [34, 23, 17, 91, 99, 87, 52, 51, 19, 5, 2, 100, 66, 45, 32]
#GRUPO SELECIONADO PELO LDA
#g = [17, 23, 91, 76, 92, 78, 66, 34, 93, 89, 70, 29, 16, 80, 54, 53, 51, 43, 30, 20, 18, 10, 99, 90, 86]
#g = [17, 23, 91, 76, 92, 78, 66, 34, 93, 89, 70, 29, 16, 80, 54]

#TOP 10
#g = [34, 23, 17, 91, 99, 51, 66, 80, 76, 54, 29]


#g = [34, 23, 17, 91, 99, 87, 52, 51, 19, 5, 2, 100, 66, 45, 32, 22, 82, 80, 76, 71, 69, 61, 54, 37, 29]
NOMES = g
#for i in range(1,101):
#    NOMES.append(i)

def plot_confusion_matrix(cm, title='Matriz de Confusao', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(NOMES))
    plt.xticks(tick_marks, NOMES, rotation=45)
    plt.yticks(tick_marks, NOMES)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


n = 10
y = []
labels = []
retirado = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
a = 0
matriz = np.zeros((n,n))
grupo = []
#%%%%%%%%%%%%%%%%%%%%%%%SELECIONANDO A BASE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
baselibras = np.loadtxt("base102.txt", delimiter = ',')
#baselibras= np.float32(baselibras)
print(np.shape(baselibras))
base = []

for j in range(n*15):
    base.append(baselibras[j][0:50])

#%%%%%%%%%%%%%%%%%%CRIANDO O CONJUNTO DE TREINAMENTO %%%%%%%%%%%%%%%%%%%%%%%
for i in range(n):
    for j in range(14):
        y.append(i)

#for i in range(15):
for j in range(n):
    labels.append(j)

for r in retirado:
    x = []
    ret = []
    for j in range(n*15):
        if j == r:
            ret.append(r)
            r = r + 15

        else:
            x.append(base[j][0:50])


# %%%%%%%%%%%%%%%%%%%%%%%%%%TREINANDO%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    clf=LDA()
    clf.fit(x, y)
    


#%%%%%%%%%%%%%%%%%%%%%%CRIANDO O CONJUNTO DE TESTE%%%%%%%%%%%%%%%%%%%%%%
    xteste = []
    
    for i in ret:
        xteste.append(base[i][0:50])

#%%%%%%%%%%%%%%%%%%%%%%%%%%TESTANDO%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    a = clf.score(xteste, labels)
    b = clf.predict(xteste)
    cm = confusion_matrix(labels,b)
    cm = np.asarray(cm)
    matriz = matriz + cm
    grupo.append(a)
    
np.set_printoptions(precision=0)
np.savetxt("matriz102lda.txt",matriz)
plt.figure()
plot_confusion_matrix(matriz)
plt.show()
print('Acurácia media do grupo: ', np.mean(grupo))
print('Desvio padrão do grupo: ', np.std(grupo))
