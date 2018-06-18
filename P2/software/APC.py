# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:40:54 2018

@author: Santi
"""

import numpy as np
from sklearn.model_selection import KFold
import time

class individuo:
    def __init__(self,_pesos,X_train,Y_train,X_test,Y_test):
        self.pesos = _pesos
        clasificador = clasificador1nn(X_train, Y_train, X_test, self.pesos)  
        f = func_global(clasificador,Y_test,self.pesos)
        self.funcion = f
    def __gt__(self,individuo):
        return self.funcion > individuo.funcion
    def __le__(self,individuo):
        return self.funcion <= individuo.funcion
        
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
    
    w = np.array(w)    
    x = []
    y = []    
    for i in range(0,len(w)):
        x.append(np.array(w[i,0:len(w[i])-1]))    
        y.append(np.array(w[i,len(w[i])-1:len(w[i])]))

    return w

#Este método normaliza en el intervalo [0,1] y devuelve X e Y. El parámetro es el conjunto de datos COMPLETO
def normalizar(X):
    Y = X[::,len(X[0])-1:len(X[0])]
    X = X[::,0:len(X[0])-1]

    maximo = np.max(Y)

    Y[Y/maximo<1.0] = 0.0
    Y[Y/maximo==1.0]=1.0

    total = np.max(X[::,0:1]) - np.min(X[::,0:1])
    X_normalizada = np.array((X[::,0:1] - np.min(X[::,0:1]))/total)
 
    for i in range(2,len(X[0])):
        total = np.max(X[::,i-1:i]) - np.min(X[::,i-1:i])
        columna = np.array((X[::,i-1:i] - np.min(X[::,i-1:i]))/total)
        X_normalizada = np.concatenate((X_normalizada, columna), axis=1)  
        
    return X_normalizada,Y

# En esta función, X1 son los datos de entrenamiento y X2 la muestra que quiero clasificar
def dist(X1,X2,w):
    return np.sum(w*np.abs(X2-X1))

def distancia(X1,X2):
    return np.sum(np.abs(X2-X1))

#Clasifica un valor del conjunto de test a partir del conjunto de train.
def clasificador1nn(X_train,Y_train,X_test, w):
    cmin = Y_train[0]
    c_out = []
    w_ = np.copy(w)
    w_ [w_ < 0.2] = 0
    for j in range (len(X_test)):
        dmin = dist(X_train[0],X_test[j],w_)        
        for i in range(0,len(X_train)):
            d = dist(X_train[i],X_test[j],w_)
            if(d < dmin and d > 0.0):
                cmin,dmin = Y_train[i],d
        c_out.append(cmin)   
    return c_out        

def funcion_global(tasa_clas,tasa_red):
    return 0.5*tasa_clas + 0.5*tasa_red

def tasa_clasificacion(clasificacion,Y):
    contador = 0
    for i in range(len(clasificacion)):
        if (clasificacion[i]==Y[i]):
            contador = contador + 1

    return (contador/len(Y))*100

def tasa_red(w):
    contador = 0 
    for i in range(len(w)):
        if(w[i] < 0.2):
            contador = contador + 1
            
    return (contador/len(w))*100

def genera_poblacion(X_train,n):
    poblacion = []
    for i in range(0,n):
        poblacion.append(genera_solucion_inicial(len(X_train[0])))
    
    return poblacion    

def generar_poblacion(X_train,Y_train,X_test,Y_test,n):
    poblacion = []
    for i in range(0,n):
        sol_ini = genera_solucion_inicial(len(X_train[0]))
        ind = individuo(sol_ini,X_train,Y_train,X_test,Y_test)
        poblacion.append(ind)
    return poblacion  
       
def cruce(cromosoma1,cromosoma2,alfa,X_train,Y_train,X_test,Y_test):
    descendientes = []
    
    for j in range(0,2):
        hijo = []
        for i in range(len(cromosoma1)):
            c_max = np.max([cromosoma1[i],cromosoma2[i]])
            c_min = np.min([cromosoma1[i],cromosoma2[i]])
            I = c_max - c_min            
            valor = np.random.uniform(c_min-I*alfa,c_max+I*alfa)
            if(valor < 0):
                valor = 0
            if(valor > 1):
                valor = 1
                
            hijo.append(valor) 
        
        h = individuo(hijo,X_train,Y_train,X_test,Y_test)        
        descendientes.append(h)
    return descendientes    
   
def torneo_binario(poblacion):
    c1 = np.random.randint(0,len(poblacion))
    c2 = np.random.randint(0,len(poblacion))
    while(c2==c1):
        c2 = np.random.randint(0,len(poblacion))

    i1 = poblacion[c1]
    i2 = poblacion[c2]
    if(i1.funcion > i2.funcion):
        return i1
    else:
        return i2

def AM(generaciones,tasa,X_train,Y_train,x_test,y_test,tama,tip):
    evalua = 0
    poblacion = generar_poblacion(X_train,Y_train,x_test,y_test,tama)
    gene = 0
    while(evalua < 1000):
        padres = []
        np.random.shuffle(poblacion)
        
        #SELECCIÓN
        for i in range(np.int(0.7*len(poblacion))):        
            padres.append(torneo_binario(poblacion))
        
        #CRUCE
        hijos = []
        p = 0

        while( p < (len(padres) - 2)):
            cruzados1 = cruce_aritmetico(padres[p].pesos,padres[p + 1].pesos,X_train,Y_train,X_test,Y_test)
            cruzados2 = cruce_aritmetico(padres[p + 1].pesos,padres[p + 2].pesos,X_train,Y_train,X_test,Y_test)                  
            evalua = evalua + 2
            hijos.append(cruzados1)
            hijos.append(cruzados2)
            p = p + 2
                
        
        #MUTACIÓN
        hijos_mutados = []
        for m in range(len(hijos)):
            for n in range(len(hijos[m].pesos)):
                numero_aleatorio = np.random.random(1)
                if(numero_aleatorio < 0.001):
                    v = np.random.normal()
                    valor = np.random.randint(0,len(hijos[0].pesos))
                    hijos[m].pesos[valor] = hijos[m].pesos[valor] + v 
            
                    if(hijos[m].pesos[valor] > 1):
                        hijos[m].pesos[valor]=1
            
                    if(hijos[m].pesos[valor] < 0):
                        hijos[m].pesos[valor]=0
                
            hijos_mutados.append(hijos[m])   
         
        nueva_poblacion = hijos_mutados.copy()
        np.sort(poblacion)
        nueva_poblacion.append(poblacion[0])

        while(len(nueva_poblacion) < tama):
            v = np.random.randint(1,len(poblacion))
            nueva_poblacion.append(poblacion[v])
       
       
        if(tip == 'M'):
            np.sort(nueva_poblacion)
            for i in range(np.int(tasa*len(poblacion))):
                nueva_poblacion[i] = individuo(BL(X_train,Y_train,x_test,y_test,2,nueva_poblacion[i].pesos),X_train,Y_train,x_test,y_test)
                
        else:    
            if(gene % generaciones == 0):
                for i in range(len(nueva_poblacion)):
                    valor = np.random.random(1)
                    if(valor < tasa):
                        nueva_poblacion[i] = individuo(BL(X_train,Y_train,x_test,y_test,2,nueva_poblacion[i].pesos),X_train,Y_train,x_test,y_test)
                    
        poblacion = nueva_poblacion.copy()
        gene = gene + 1
                
    np.sort(poblacion)
    return poblacion[0].pesos             


def AGE(X_train,Y_train,x_test,y_test,tama,tipo):
    evalua = 0
    poblacion = generar_poblacion(X_train,Y_train,x_test,y_test,tama)

    while(evalua < 1000):
        padres = []
        np.random.shuffle(poblacion)
        #SELECCIÓN
        if(tipo == "BLX"):
            for i in range(2):        
                padres.append(torneo_binario(poblacion))    
        else:
            for i in range(4):        
                padres.append(torneo_binario(poblacion))               
        #CRUCE
        hijos = []
        p = 0

        if(tipo == "BLX"):
           cruzados = cruce(padres[0].pesos,padres[1].pesos,0.3,X_train,Y_train,X_test,Y_test)
           evalua = evalua + 2            
           hijos.append(cruzados[0])
           hijos.append(cruzados[1])
           
        else:
           cruzados1 = cruce_aritmetico(padres[0].pesos,padres[1].pesos,X_train,Y_train,X_test,Y_test)   
           cruzados2 = cruce_aritmetico(padres[2].pesos,padres[3].pesos,X_train,Y_train,X_test,Y_test)   
           evalua = evalua + 2            
           hijos.append(cruzados1)
           hijos.append(cruzados2)           
           
        #MUTACIÓN
        hijos_mutados = []
        for m in range(len(hijos)):
            for n in range(len(hijos[m].pesos)):
                numero_aleatorio = np.random.random(1)
                if(numero_aleatorio < 0.001):
                    v = np.random.normal()
                    valor = np.random.randint(0,len(hijos[0].pesos))
                    hijos[m].pesos[valor] = hijos[m].pesos[valor] + v 
            
                    if(hijos[m].pesos[valor] > 1):
                        hijos[m].pesos[valor]=1
            
                    if(hijos[m].pesos[valor] < 0):
                        hijos[m].pesos[valor]=0
                
            hijos_mutados.append(hijos[m])
         
        eliminado1 = False
        eliminado2 = False
        minimo = np.min(poblacion)
        
        if(minimo < np.min(hijos_mutados)):
            poblacion.remove(minimo)            
            eliminado1 = True  
            
        minimo = np.min(poblacion)
        if(minimo < np.max(hijos_mutados)):
            poblacion.remove(minimo)   
            eliminado2 = True

        if(eliminado1):
            poblacion.append(np.min(hijos_mutados))
            eliminado1 = False
        if(eliminado2):
            poblacion.append(np.max(hijos_mutados)) 
            eliminado2 = False

    return np.max(poblacion).pesos   
    
def AGG(X_train,Y_train,x_test,y_test,tama,tipo):
    
    evalua = 0
    poblacion = generar_poblacion(X_train,Y_train,x_test,y_test,tama)
    
    while(evalua < 1000):
        padres = []
        np.random.shuffle(poblacion)
        #SELECCIÓN
        for i in range(np.int(0.7*len(poblacion))):        
            padres.append(torneo_binario(poblacion))
        
        #CRUCE
        hijos = []
        p = 0

        if(tipo == "BLX"):
           # print(len(padres))        
            while( p < (len(padres)-1)):
                cruzados = cruce(padres[p].pesos,padres[p + 1].pesos,0.3,X_train,Y_train,X_test,Y_test)
                evalua = evalua + 2

                hijos.append(cruzados[0])
                hijos.append(cruzados[1])
                p = p + 2
                
        else:  
            while( p < (len(padres) - 2)):
                cruzados1 = cruce_aritmetico(padres[p].pesos,padres[p + 1].pesos,X_train,Y_train,X_test,Y_test)
                cruzados2 = cruce_aritmetico(padres[p + 1].pesos,padres[p + 2].pesos,X_train,Y_train,X_test,Y_test)
                
                evalua = evalua + 2
             
                hijos.append(cruzados1)
                hijos.append(cruzados2)
                p = p + 2
                
        
        #MUTACIÓN
        hijos_mutados = []
        for m in range(len(hijos)):
            for n in range(len(hijos[m].pesos)):
                numero_aleatorio = np.random.random(1)
                if(numero_aleatorio < 0.001):
                    v = np.random.normal()
                    valor = np.random.randint(0,len(hijos[0].pesos))
                    hijos[m].pesos[valor] = hijos[m].pesos[valor] + v 
            
                    if(hijos[m].pesos[valor] > 1):
                        hijos[m].pesos[valor]=1
            
                    if(hijos[m].pesos[valor] < 0):
                        hijos[m].pesos[valor]=0
                
            hijos_mutados.append(hijos[m])   
         
        nueva_poblacion = hijos_mutados.copy()
        np.sort(poblacion)
        nueva_poblacion.append(poblacion[0])

        while(len(nueva_poblacion) < tama):
            v = np.random.randint(1,len(poblacion))
            nueva_poblacion.append(poblacion[v])
       
        poblacion = nueva_poblacion.copy()
    
    np.sort(poblacion)
    return poblacion[0].pesos         
        
def cruce_aritmetico(cromosoma1,cromosoma2,X_train,Y_train,X_test,Y_test):
    descendiente = []
    for i in range(len(cromosoma1)):
        descendiente.append((cromosoma1[i]+cromosoma2[i])/2)
    
    nuevo_descendiente = individuo(descendiente,X_train,Y_train,X_test,Y_test)    
    return nuevo_descendiente
   
    
#Tipo será si el vecino es amigo(valor 1) o enemigo(0)
def buscar_vecino(x,y,elemento,tipo):
    x = np.concatenate((x,y), axis=1)
    actual = x[elemento]
    x = np.concatenate((x[0:elemento],x[elemento+1:len(x)]))
    p = 0
    vecino_cercano = x[p]

    while((tipo+1)%2 and (vecino_cercano[len(x[p])-1]==actual[len(actual)-1])):
        vecino_cercano = x[p]
        p = p + 1    
    
   
    if(tipo==1):
        for i in range(1,len(x)):
            if(x[i][len(x[i])-1] == actual[len(actual)-1]):
                if(distancia(x[i],actual)<=distancia(vecino_cercano,actual)):
                    vecino_cercano = x[i]                
    if(tipo==0):
        for i in range(1,len(x)):
            if(x[i][len(x[i])-1] != actual[len(actual)-1]):
                if(distancia(x[i],actual)<=distancia(vecino_cercano,actual)):
                    vecino_cercano = x[i]
              
    return vecino_cercano[0:len(vecino_cercano)-1],vecino_cercano[len(vecino_cercano)-1:len(vecino_cercano)]

def genera_solucion_inicial(n):
    return np.random.uniform(size=n)

def generar_vecinos(solucion):
    solucion_actual = solucion.copy()
    soluciones = []
    for i in range(len(solucion_actual)):
        v = np.random.normal()
        solucion_actual[i] = solucion_actual[i] + v
        if(solucion_actual[i] > 1):
           solucion_actual[i]=1
        if(solucion_actual[i] < 0):
           solucion_actual[i]=0
           
        soluciones.append(solucion_actual)
        solucion_actual = solucion.copy()
    return soluciones    

def func_global (clasificacion,Y,w):
    return 0.5*tasa_clasificacion(clasificacion,Y)+0.5*tasa_red(w)
    
def greedy_RELIEF(x,y):
    w = np.zeros(len(x[0]))
    
    for i in range(len(x)):
    
        enemigo_x,enemigo_y = buscar_vecino(x,y,i,0) 
        amigo_x,amigo_y = buscar_vecino(x,y,i,1)       
        w = w + np.abs(x[i]-enemigo_x) - np.abs(x[i]-amigo_x)
    
    w_m = np.max(w)
    
    for j in range(0,len(w)):
    
        if(w[j] < 0):
            w[j] = 0.0
            
        else:
            w[j] = (w[j]/w_m)
            
    return w

def BL(x,y,x_test,y_test,n,sol_inicial):
   
    #if(sol_inicial == 0):
    #    sol_inicial = genera_solucion_inicial(len(x[0]))
    
    sol_ini = sol_inicial    
    mejor_sol = sol_ini
    
    clasificador = clasificador1nn(x,y,x_test,mejor_sol)

    tc_mejor = tasa_clasificacion(clasificador,y_test)
    tr_mejor = tasa_red(mejor_sol)
    
    evaluacion = 0
    num_vecinos = 0

    while(num_vecinos <n*len(x[0]) and evaluacion <1000):
        soluciones_vecinas = generar_vecinos(mejor_sol) 
    
        for sol in soluciones_vecinas:
            tr = tasa_red(sol)
            clasificador = clasificador1nn(x,y,x_test,sol)
            evaluacion = evaluacion + 1
            
            tc = tasa_clasificacion(clasificador,y_test)
            num_vecinos = num_vecinos + 1

            if(funcion_global(tc,tr) > funcion_global(tc_mejor,tr_mejor)):
                num_vecinos = 0
                mejor_sol = sol
                tc_mejor = tc 
                tr_mejor = tr
                break

    return mejor_sol
############################################################################
#PROGRAMA PRINCIPAL
############################################################################

#Comentar y descomentar para elegir el fichero de datos
    
X = np.array(leerDatos("ozone-320.arff"))
#X = np.array(leerDatos("parkinsons.arff"))
#X = np.array(leerDatos("spectf-heart.arff"))
X,Y = normalizar(X)


test_ind = []

kf = KFold(n_splits=5)
kf.get_n_splits(X)

X_train,Y_train = [],[]
X_test,Y_test = [],[]
tasas_clas = []
tasas_red = []
tiempo = []
funciones = []
itera = 1
for train_index, test_index in kf.split(X):
    #    print("TRAIN:",train_index,"TEST:", test_index)
    n_train = []
    n_train2 = []
    X_train = []
    test_ind = []
    test_ind.append(test_index)
    #print(X_test)    
    total = X.shape[0]-1
   
    test_ind = np.array(test_ind)
    
    for i in range(0,total):
        if(i not in test_ind):
            n_train.append(i)    
        else:
            n_train2.append(i)
            
    X_train,Y_train = X[n_train],Y[n_train]
    X_test,Y_test = X[n_train2],Y[n_train2]

    print("Conjunto ",itera)
    itera = itera + 1
    
    t1 = time.time()
    
    #w = np.ones(len(X_train[0]))
    #w = greedy_RELIEF(X_train,Y_train)
    #sol_temp = genera_solucion_inicial(len(X_train[0]))
    #w = BL(X_train,Y_train,X_test,Y_test,20,sol_temp)
    
    
    #w = AGG(X_train,Y_train,X_test,Y_test,30,"BLX")
    w = AGG(X_train,Y_train,X_test,Y_test,30,"AC")
    #w = AGE(X_train,Y_train,X_test,Y_test,30,"BLX")
    #w = AGE(X_train,Y_train,X_test,Y_test,30,"AC")

    #w = AM(10,1.0,X_train,Y_train,X_test,Y_test,30,'-')
    #w = AM(10,0.1,X_train,Y_train,X_test,Y_test,30,'-')
    #w = AM(10,0.1,X_train,Y_train,X_test,Y_test,30,'M')

    t2 = time.time()
    tiempo.append(t2-t1)

    clasificador = clasificador1nn(X_train,Y_train,X_test,w)
    tc = tasa_clasificacion(clasificador,Y_test)
    tr = tasa_red(w)
    tasas_clas.append(tc)
    tasas_red.append(tr)
    funciones.append(funcion_global(tr,tc))
    #print("",tc)
    #print("tasa_red: ",tr)
    #print("func_global: ",funcion_global(tr,tc))

    n_train.clear()
    n_train2.clear()
    #print(XY_test)

print("tiempos\n")
for t in tiempo:
    print(t)
    
print("\ntasa de clasificacion\n")
for p in tasas_clas:
    print(p)    
    
print("\ntasa de reducción\n")
for q in tasas_red:
    print(q)        

print("\nfuncion solucion\n")
for s in funciones:
    print(s)       
    