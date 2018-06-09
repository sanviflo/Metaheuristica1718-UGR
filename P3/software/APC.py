# -*- coding: utf-8 -
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

def genera_vecino(solucion):
    v = np.random.normal()
    Z = np.zeros(len(solucion))
    Z[np.random.randint(0,len(solucion))] = v
    return solucion + Z
    
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

def mutacion_brusca(w):
    lista = []
    for i in range(np.int(0.1*(len(w)))):
        lista.append(np.random.randint(0,len(w)))

    w [lista] = np.random.normal()
    #for i in lista:
    #    w[i] = np.random.normal()
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

def ILS (X_train,Y_train,X_test,Y_test):
    
    sol =  genera_solucion_inicial(len(X_train[0]))
    clasificacion = clasificador1nn(X_train,Y_train,X_test,sol)    
    coste = func_global (clasificacion,Y_test,sol)
    porcentaje = 0
    
    mejor_sol = sol
    mejor_coste = coste
    
    print("[",porcentaje,"% ]")

    for i in range(15):
        sol_nueva = BL(X_train,Y_train,X_test,Y_test,5,mejor_sol)
        clasificacion2 = clasificador1nn(X_train,Y_train,X_test,sol_nueva)
        coste2 = func_global (clasificacion2,Y_test,sol_nueva)

        if(coste2 > mejor_coste):
            mejor_sol = sol_nueva
            mejor_coste = coste2        
         
            sol = mutacion_brusca(mejor_sol)    
            mejor_sol = sol
            clasificacion = clasificador1nn(X_train,Y_train,X_test,mejor_sol)    
            mejor_coste = func_global (clasificacion,Y_test,mejor_sol)
        
        if(i % 2 == 0):
           porcentaje = porcentaje + 7
           print("[",porcentaje,"% ]")                
    
    print("[100%]")
    return mejor_sol

def seleccionar_3_padres(poblacion,i):
    padre1 = np.random.randint(0,len(poblacion))
    while(padre1 == i):
        padre1 = np.random.randint(0,len(poblacion))

    padre2 = np.random.randint(0,len(poblacion))
    while(padre2 == i or padre2 == padre1):
        padre2 = np.random.randint(0,len(poblacion))                

    padre3 = np.random.randint(0,len(poblacion))
    while(padre3 == i or padre3 == padre1 or padre3 == padre2):
        padre3 = np.random.randint(0,len(poblacion))    

    return np.copy(poblacion[padre1]),np.copy(poblacion[padre2]),np.copy(poblacion[padre3])

def DE_rand(X_train,Y_train,X_test,Y_test,n):
    poblacion = genera_poblacion(X_train,n)
    evaluacion = []
    evalua = 0
    mejor_sol,mejor_func = 0,0
    porcentaje = 0
    size = 15000
    
    print("[",porcentaje,"% ]")
    
    for i in range(n):
        clasificacion = clasificador1nn(X_train,Y_train,X_test,poblacion[i])
        fit = func_global(clasificacion,Y_test,poblacion[i])
        evaluacion.append(fit)
        evalua = evalua + 1
        if(fit > mejor_func):
            mejor_func = fit
            mejor_sol = poblacion[i]
        
    while(evalua < size):
        for i in range(0,len(poblacion)):
            padre1,padre2,padre3 = seleccionar_3_padres(poblacion,i)           
            offspring = np.zeros(len(poblacion[0]))

            for k in range(len(poblacion[0])):
                if(np.random.rand() < 0.5):
                    offspring[k] = padre1[k] + 0.5*(padre2[k] - padre3[k])
                else:
                    offspring[k] = poblacion[i][k]
                    
            clasificacion2 = clasificador1nn(X_train,Y_train,X_test,offspring)
            funcion = func_global(clasificacion2,Y_test,offspring)
            evalua = evalua + 1
            
            if(funcion > evaluacion[i]):
                poblacion[i] = offspring
                evaluacion[i] = funcion

            if(funcion > mejor_func):
                mejor_func = funcion
                mejor_sol = offspring

        if(evalua % (size/100) == 0):
           porcentaje = porcentaje + 1
           print("[",porcentaje,"% ]")
    
    if(evalua>=size):       
        if(evalua % (size/100) == 0):
           porcentaje = porcentaje + 1
           print("[100 % ]")
    
    return mejor_sol

def DE_current(X_train,Y_train,X_test,Y_test,n):
    poblacion = genera_poblacion(X_train,n)
    evaluacion = []
    evalua = 0
    mejor_sol,mejor_func = 0,0
    porcentaje = 0
    size = 15000
    
    print("[",porcentaje,"% ]")
    
    for i in range(n):
        clasificacion = clasificador1nn(X_train,Y_train,X_test,poblacion[i])
        fit = func_global(clasificacion,Y_test,poblacion[i])
        evaluacion.append(fit)
        evalua = evalua + 1
        if(fit > mejor_func):
            mejor_func = fit
            mejor_sol = poblacion[i]
        
    while(evalua < size):
        for i in range(0,len(poblacion)):
            padre1,padre2,padre3 = seleccionar_3_padres(poblacion,i)           
            offspring = np.zeros(len(poblacion[0]))

            for k in range(len(poblacion[0])):
                if(np.random.rand() < 0.5):
                    offspring[k] = poblacion[i][k] + 0.5*(mejor_sol[k] - poblacion[i][k]) + 0.5*(padre1[k] - padre2[k])
                else:
                    offspring[k] = poblacion[i][k]
                    
            clasificacion2 = clasificador1nn(X_train,Y_train,X_test,offspring)
            funcion = func_global(clasificacion2,Y_test,offspring)
            evalua = evalua + 1
            
            if(funcion > evaluacion[i]):
                poblacion[i] = offspring
                evaluacion[i] = funcion

            if(funcion > mejor_func):
                mejor_func = funcion
                mejor_sol = offspring

        if(evalua % (size/100) == 0):
           porcentaje = porcentaje + 1
           print("[",porcentaje,"% ]")
    
    if(evalua>=size):       
        if(evalua % (size/100) == 0):
           porcentaje = porcentaje + 1
           print("[100 % ]")
    
    return mejor_sol
    

def ES (X_train,Y_train,X_test,Y_test):

    Tf = 0.01
    M = 15000 
    n_exitos = 0
    max_vecinos = (len (X_train[0]))
    max_exitos = 0.1 * max_vecinos
    sol =  genera_solucion_inicial(len(X_train[0]))
    clasificacion = clasificador1nn(X_train,Y_train,X_test,sol)
    porcentaje = 0
    coste = func_global (clasificacion,Y_test,sol)

    T0 = (0.3*coste)/(-np.log(0.3)) 
    T = T0
    B = (T0 - Tf)/(M*T0*Tf)
    
    best_solution = sol
    best_coste = coste    
    print("[",porcentaje,"% ]")
    
    for i in range (0, M):
        n_exitos = 0
        for j in range (1, max_vecinos):
            s = genera_vecino(sol)
            clasificacion2 = clasificador1nn(X_train,Y_train,X_test,s)
            coste2 = func_g558lobal (clasificacion,Y_test,s)            
            
            dif = coste - coste2
            
            if(dif < 0 or np.random.uniform() < np.exp(-dif/T)):
                n_exitos = n_exitos + 1
                sol = s
                coste = coste2
                if(coste > best_coste):
                    best_coste = coste
                    best_solution = sol
                    
            if(n_exitos >= max_exitos):
                break

        T_new = T / (1 + B*T)
        T = T_new

        print(i)
        if(evalua % (M/100) == 0):
           porcentaje = porcentaje + 1
           print("[",porcentaje,"% ]")
    
        if(n_exitos == 0):
            break
        
    return best_solution
############################################################################
#PROGRAMA PRINCIPAL
############################################################################

#Comentar y descomentar para elegir el fichero de datos
    
#X = np.array(leerDatos("ozone-320.arff"))
X = np.array(leerDatos("parkinsons.arff"))
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
   
#    w = ILS(X_train,Y_train,X_test,Y_test)
#    w = ES(X_train,Y_train,X_test,Y_test)
#    w = DE_rand(X_train,Y_train,X_test,Y_test,50)
    w = DE_current(X_train,Y_train,X_test,Y_test,50)
    

    t2 = time.time()
    tiempo.append(t2-t1)

    clasificador = clasificador1nn(X_train,Y_train,X_test,w)
    tc = tasa_clasificacion(clasificador,Y_test)
    tr = tasa_red(w)
    tasas_clas.append(tc)
    tasas_red.append(tr)
    funciones.append(funcion_global (tr,tc))
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
    