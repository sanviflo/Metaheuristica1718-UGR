# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:40:54 2018

@author: Santi
"""

import numpy as np
from random import shuffle
from scipy.spatial import distance
import time 
from sklearn.model_selection import LeaveOneOut
#Lee los datos del fichero ubicado en el directorio path
def leerDatos(path):
    w = []
    w2 = []
    lee_datos = 0
    archivo = open(path)
    w.clear()
    for linea in archivo.readlines():    
        l = linea.split(",")
        
        if (lee_datos==1):
            for i in l:
               w2.append(np.float(i))
            
            w.append(w2.copy()) 
            w2.clear()

        if ("@data" in l[0]):
            lee_datos=1
    return w

#Devuelve la clase a la que pertenece un elemento
def clase (datos, elem):
    return datos[elem][len(datos[elem])-1]

def minimo (datos,atributo):
    i = 0
    minimo = datos[i][atributo]
    for i in (1,len(datos)-1):
        if datos[i][atributo] < minimo:
           minimo = datos [i][atributo] 
    return minimo

def maximo (datos,atributo):
    i = 0
    maximo = datos[i][atributo]
    for i in (1,len(datos)-1):
        if datos[i][atributo] > maximo:
           maximo = datos [i][atributo] 
    return maximo
    
#Normaliza los atributos de un dato
def normalizarDato(lista,linea):
    datos = lista[linea].copy()
    for i in range(0,len(datos)):
        datos[i] = (datos[i] - minimo(lista,i))/(maximo(lista,i) - (minimo(lista,i)))
    return datos

#Normaliza todos los datos
def normalizacion(lista):
    vector2 = []

    for i in range(0,len(vector)):
        vector2.append(normalizarDato(vector,i))
        
    return vector2
    
#Parte del programa que clasifica los datos según el esquema del seminario
    
def clasificador1nn(train1,test,pesos):
    train = train1.copy()
    cmin = clase(train,0)
    dmin = distancia(train[0],test,pesos)
    
    for i in range(1,len(train)):
       d = distancia(train[i],test,pesos)  
       if (d<dmin):
           cmin = clase(train,i)
           dmin = d
    return cmin       

#distancia ponderada
def distancia(train,test,pesos):
    dist = 0
    for i in range (len(train)):    
        dist = np.sqrt(dist + pesos[i]*np.power((train[i]-test[i]),2))
    return dist    
    
#nº de valores de wi < 0.2
def n_valores(W):
    cuenta_valores = 0
    for i in range (0,len(W)):
        if (W[i] < 0.2):
            cuenta_valores = cuenta_valores+1
    return cuenta_valores


#Tasa de reducción
def tasa_red (W):
    return (n_valores(W)/len(W))*100

#Tasa de clasificación
def tasa_clas (W,clasificacion):
    cuenta_aciertos = 0
    for i in range(0,len(W)):
        if (clase(W,i) == clasificacion[i]):
           cuenta_aciertos = cuenta_aciertos + 1
    return (cuenta_aciertos/len(W))*100

#Busca el enemigo del valor dentro del train cuya distancia sea menor
def buscarEnemigoCercano(train,valor):
    i = 0
    dist = 0
    c = 0
    elem = []
    
    while(train[i]==valor or clase(train,i)==valor[len(valor)-1]):
        i = i + 1
        
    dist = distance.euclidean(valor,train[i])
    c = clase(train,i)    
    elem = train[i]
    
    for j in range(i+1,len(train)):
        if(train[j]!=valor and clase(train,j)!=valor[len(valor)-1] and distance.euclidean(valor,train[j]) < dist):
            dist = distance.euclidean(valor,train[j])
            c = clase(train,j)
            elem = train[j]
    
    return elem            
        
#Busca el enemigo del valor dentro del train cuya distancia sea menor
def buscarAmigoCercano(train,valor):
    i = 0
    dist = 0
    c = 0
    elem = []
    
    while(train[i]==valor or clase(train,i)!=valor[len(valor)-1]):
        i = i + 1
        
    dist = distance.euclidean(valor,train[i])
    c = clase(train,i)    
    elem = train[i]
    
    for j in range(i+1,len(train)):
        if(train[j]!=valor and clase(train,j)==valor[len(valor)-1] and distance.euclidean(valor,train[j]) < dist):
            dist = distance.euclidean(valor,train[j])
            c = clase(train,j)
            elem = train[j]
    
    return elem   

#función que evalúa como de buena es una solución con un alfa = 0.5 por defecto
def func_evalua(tasa_red,tasa_clas,alfa=0.5):
    return alfa*tasa_clas + (1-alfa)*tasa_red

#Algoritmo Greedy que me asigna los pesos a las características de mi conjunto de datos
def greedy(train):
    #Vector solución
    W = [0 for i in range(len(train[0]))]
    #print(W)

    for e_i in train:
    
        e_e = buscarEnemigoCercano(train,e_i)
        e_a = buscarAmigoCercano(train,e_i)
        
        for j in range(0,len(W)):
            W[j] = W[j]+np.abs(e_i[j] - e_e[j])-np.abs(e_i[j] - e_a[j])
    
    w_m = max(W)    
    for i in range(0,len(W)):
        if(W[i] < 0):
            W[i] = 0
        else:
            W[i] = W[i]/w_m
    return W        

#Divide nuestros datos en 5 trozos manteniendo la distribución equilibrada
def divide_datos(train):
    train1 = []
    train2 = []

    for i in range (0,len(train)):
        if(clase(train,i)==1):
            train1.append(train[i])
        else:
            train2.append(train[i])

    particiones1 = np.int(len(train1)/5)
    particiones2 = np.int(len(train2)/5)
    
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    
    for j in range(particiones1):
        t1.append(train1[j])
        t2.append(train1[particiones1+j])
        t3.append(train1[2*particiones1+j])
        t4.append(train1[3*particiones1+j])
        t5.append(train1[4*particiones1+j])            
    for j in range(particiones2):
        t1.append(train2[j])
        t2.append(train2[particiones2+j])
        t3.append(train2[2*particiones2+j])
        t4.append(train2[3*particiones2+j])
        t5.append(train2[4*particiones2+j])        

    return t1,t2,t3,t4,t5

#Este método genera un vecino modificando dentro de la distribución uno de las características
def genera_vecino(solucion,pos):
    v = np.random.normal(0.0,0.3,1)
    s = solucion.copy()
    
    s[pos] = (np.float(solucion[pos] + v))
    if(s[pos] < 0.0):
        s[pos] = 0.0
    if(s[pos] > 1.0):
        s[pos] = 1.0                
    
    return s    

# Algoritmo de busqueda local
def busqueda_local(datos):
    sol_ini = [np.random.rand(1) for i in range(len(datos[0]))]
    sol_actual = sol_ini.copy()
    sol_mejor = sol_actual.copy()
    clasif = []
    encuentra = 0
    iteracion = 0
    
    while (iteracion < 15000 and encuentra<(20*len(sol_ini))):
        iteracion = iteracion + 1
        for i in range(len(sol_actual)):
            solucion_vecina = genera_vecino(sol_actual,i)
            sol_actual = solucion_vecina
        
            for i in range(len(datos)):
                clasif.append(clasificador1nn(datos,datos[i],sol_actual))

            tasa_acierto = tasa_clas(datos,clasif)
            tasa_reduccion = tasa_red(sol_actual)
            tasa_reduccion2 = tasa_red(sol_mejor)
        
            if(func_evalua(tasa_reduccion,tasa_acierto) > func_evalua(tasa_reduccion2,tasa_acierto)):
                sol_mejor = sol_actual
                encuentra = 0
            else:
                encuentra = encuentra + 1
    
    return sol_mejor        

############################################################################
#PROGRAMA PRINCIPAL
############################################################################

#Comentar y descomentar para elegir el fichero de datos
    
#vector = leerDatos("parkinsons.arff")
vector = leerDatos("ozone-320.arff")
#vector = leerDatos("spectf-heart.arff")


train = normalizacion(vector)
t1,t2,t3,t4,t5 = divide_datos(train)

#Cambiar el t entre t1 y t5
t = t1

train_data = []
test_data = []
q = []
n_train = 0.8*len(t)

shuffle(t)

for j in range(len(t)): 
    if(j<=n_train):
        train_data.append(t[j])
    else:
        test_data.append(t[j])        


#print("Datos de entrenamiento:\n",len(train_data))
#print("\nDatos de prueba:\n",len(test_data))

for i in range(len(train_data[0])):
    q.append(1.0)
    
p1 = time.time()
r = busqueda_local(train_data)
#r = greedy(train_data)
#r = q
p2 = time.time()

clasifica = []

for p in range(len(test_data)):
    clasifica.append(clasificador1nn(train_data,test_data[p],r))

#print(train_data)
#print(test_data)
tr = tasa_red(r)

for i in range(len(test_data)):
    tc = tasa_clas(test_data,clasifica)

print ("tiempo de ejecución: ",p2-p1)
print("Tasa de reducción: ",tr)
print("Tasa de clasificación: ",tc)
print("Función de evaluación: ", func_evalua(tr,tc))

