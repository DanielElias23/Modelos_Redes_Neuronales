kkk                                         ___________________________________
                                         |                                  |
                                         |             INDICE               |
                                         |__________________________________|

        1. REDES NEURONALES CON SKLEARN  (MLP)             "Ejemplos de redes neuronales simples con sklear"
        2. EJEMPLOS DE NN Y ANN                            "Otros ejemplos de redes neurnales y aplicaciones"
        3. GRADIENTE DESCENDIENTE                          "Gradiente y como optimiza el aprendizaje" 
        4. REGRESIONES CON KERAS                           "Muestra como es posible hacer predicciones con keras"
        5. CARGADO DE IMAGENES CON KERAS                   "Muestra diferentes formas de cargar imagenes a una red neuronal"
        6. EJEMPLO DE REDES NEURONALES                     "Algunos ejemplos de arquitectura de redes neuronales" 
        7. RETROPROPAGACION                                "Ejemplos de como funciona la retropropagacion en los modelos"
        8. OPTIMIZADORES DE GRADIENTES PART 1              "Optimizar gradientes con algunos ejemplos de clasificacion"
        9. OPTIMIZADORES DE GRADIENTES PART 2              "GridSearch y como es posible optimizar el gradiente" #**  
       10. FUNCION DE PERDIDA CATEGORICA, ENTROPIA         "Explica como usar la funcion de perdida entropica"
       11. RED NEURONAL CONVOLUCIONAL, DoG                 "Muestra ejemplos de redes convolucionales y DoG entrena bordes"
       12. CNN: STRIDE, ACTIVACION, POOLING Y PADDING      "Expliacion de algunas configuraciones de la red convolucional"
       13. CNN: FILTRO, CAPAS, APLANADO, CAMPO RECEPTIVO   "Mas configuraciones de la red convolucional"
       14. ARQUITECTURA CNN                                "Como construir una red convolucional"
       15. EJEMPLOS CNN                                    "Mas ejemplos de como construir una red convolucional"
       16. API KERAS                                       "Hay 3 formas de contruir redes neurnolas, muestra una forma API"
       17. APRENDIZAJE POR TRANSFERENCIAS                  "Aprendizaje por trasferencia, modelos preentrenados"
       18. TECNICAS DE REGULACION MODELOS DE APRENDIZAJE   "Como regular para obtener mejores resultados"
       19. EJEMPLOS DE APRENDIZAJE POR TRANSFERENCIA       "Ejemplos de como hacer este aprendizaje por trasferencia"
       20. TIPOS DE DATOS SECUENCIALES                     "Muestra las formas en la que se presentan los datos secuenciales"
       21. EMBENDDING                                      "Formas de hacer incrustaciones, tokenizar y tomar el lema"
       22. REDES NEURONALES RECURRENTES                    "Modelos de redes neuronales recurrentes"
       23. RNN: LSTM Y GRU                                 "Otras formas de redes neuronales recurrentes para evitar problemas"
       24. EJEMPLOS DE RNN                                 "Muestra algunos ejemplos de redes neuronales recurrentes"
       25. TIPOS DE AUTOCODIFICADORES                      "Todas las formas posibles que pueden hacer autocoficadores"
       26. EJEMPLOS DE AUTOCODIFICADORES                   "Ejemplos de codificadores automaticos"
       27. NEURONAS GENERADORAS ADVERSATIVAS (GAN)         "Muestra como funcionan los GAN"
       28. CNN Y GAN: DGCAN                                "Como es posible generar imagenes uniendo CNN y GAN"
       29. AUTOCODIFICADORES VARIACIONALES (VAN)           "Explica como funciona un autocodificador variacional"  #**
       30. GPU Y CPU                                       "Muestra diferencias entre los dos y como mejora CNN con GPU"
       31. APRENDIZAJE POR REFORZAMIENTO                   "Como funciona un aprendizaje de acciones y recompensas" #**
       

#######################################################################################################################

                                         #Redes Neuronales con sklearn
                                #Perceptron multicapa (MLP) Multi-layer perceptron
                                      #Es una ANN, artificial neural network 
                                     
#La clase MLPClassifier ocupa la tecnica de aprendizaje llamada retropropagacion para el entrenamiento, es un modelo
#no lineal
                                           
import warnings
warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

digits = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module1/L1/data/digits.csv")

#784 columnas que tienen cada uno pixeles, pero 1 es un label  tienen el valor del numero resultado
print(digits.head())

#labels es y
labels = digits['label']
#digits es X
digits = np.array(digits.drop('label', axis=1)).astype('float')

#Tiene 42000 imagenes que es la cantidad de filas
#digits 784, labels 1
print(digits.shape, labels.shape)

plt.figure(figsize=(12,4))
#muestra 5 imagenes aleatoriaas
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(random.choice(digits).reshape(28,28))
    plt.axis("off")
plt.show()

split = 0.7, 0.3 # train, test

#Para normalizar la data, es una variable definida
digits /= 255.0

#split es una valor concreto que multiplicado a split_ind
split_ind = int(len(digits)*split[0])

#Hace la separacion en datos de entrenamiento y prueba manual
#separacion de digits y labels
X_train, X_test, y_train, y_test = digits[:split_ind], digits[split_ind:], labels[:split_ind], labels[split_ind:]
print(X_train.shape, X_test.shape)

#para crear la capa podemos ocupar gridsearch para saber cuales son los mejores hiperparametros que puede tener

model = MLPClassifier().fit(X_train, y_train)
y_pred = model.predict(X_test)

#ocupamos acuraccy para saber la puntuacion del modleo anterior
print(f"Using MLPClassifier with the default parameter values gives an accuracy of {accuracy_score(y_pred, y_test)}")

#Vemos todas las puntuaciones por labels
print(classification_report(y_pred, y_test))

#para ocupar gridsearch en este caso usamos el solucionador predeterminado= "adam" y el activador=Relu

#GridSearchCV define una cantidad ordenada de capas, mientras que RandomizedSearchCV es aleatorio, se dice que
#randoom entrena mucho mas rapido

parameters = {'hidden_layer_sizes':[50, 200, 300],
              'alpha': [0.01, 0.1,1], 
              'max_iter': [100, 200, 500], 
              'learning_rate_init':[0.001, 0.01, 0.1, 1]}

#definimos otro modelo para ocupar gridsearch
model = MLPClassifier()
clf = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=5)
#se reduce la data por la randomizacion
clf.fit(X_train[:3000], y_train[:3000])

#Muestra cuales son los mejores parametros
print("The best parameter values found are:\n")
print(clf.best_params_)

# store the best model found in "bestmodel"
bestmodel = clf.best_estimator_
print(bestmodel)

#Ocupa enseguida el mejor modelo
y_pred = bestmodel.predict(X_test)
print(f"The accuracy score of the best model is {accuracy_score(y_test, y_pred)}\n")

#Muestra imagenes y dice caul es su label predicho
plt.figure(figsize=(12,8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    sample = random.choice(X_test)
    plt.imshow(sample.reshape(28,28))
    pred = bestmodel.predict(sample.reshape(1,-1))
    plt.title(f"Predicted as {pred}")
    plt.axis("off")

plt.tight_layout()
plt.show()


#######################################################################################################################

                                               #Ejemplos NN, ANN

import warnings
warnings.simplefilter('ignore')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Definiendo la funcion sigmoid como activador
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#Grafica la funcion
vals = np.linspace(-10, 10, num=100, dtype=np.float32)
activation = sigmoid(vals)
fig = plt.figure(figsize=(12,6))
fig.suptitle('Sigmoid function')
plt.plot(vals, activation)
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.yticks()
plt.ylim([-0.5, 1.5])
plt.show()

#Necesita entregarle x1 y x2 para que funcione
#los x1, y x2 son los a y b de la otra funion
def logic_gate(w1, w2, b):
    #una funcion que se le entregan los pesos y 1 bias
    #la funcion aplicar la funncion sigmoid, al primer peso se multiplica * valor y el segundo lo mismo + bias
    #todo junto lanza solo una entra y solo una salida
    #sigmoid input negativo, valor 0 output, input positivo, valor cercano a 1
    return lambda x1, x2: sigmoid(w1 * x1 + w2 * x2 + b)

def test(gate):
    #Esto es para ver que hace ese valor 
    for a, b in (0, 0), (0, 1), (1, 0), (1, 1):
        #entrega 3 valores en los corchetes, {a}, {b}, {np.round(date(a,b))} round debido a que entregara un decimal de
        #salida si o si, esto lo redondeara a 1 o 0
        print("{}, {}: {}".format(a, b, np.round(gate(a, b))))

#lo prueba con estos pesos
or_gate = logic_gate(20, 20, -10)
test(or_gate)
print("------")

#lo prueba con los siguientes pesos
w1 = 11
w2 = 10
b = -20
and_gate = logic_gate(w1, w2, b)
test(and_gate)

print("------")

#lo prueba con los siguientes pesos
w1 = -20
w2 = -20
b = 10
nor_gate = logic_gate(w1, w2, b)
test(nor_gate)

print("------")

#Lo prueba con los siguientes pesos
w1 = -11
w2 = -10
b = 20
nand_gate = logic_gate(w1, w2, b)
test(nand_gate)

print("------")

def xor_gate(a, b):
    c = or_gate(a, b)
    d = nand_gate(a, b)
    return and_gate(c, d)
test(xor_gate)

print("------")

#crea pesos en forma de array de 3, 4
W_1 = np.array([[2,-1,1,4],[-1,2,-3,1],[3,-2,-1,5]])
W_2 = np.array([[3,1,-2,1],[-2,4,1,-4],[-1,-3,2,-5],[3,1,1,1]])
W_3 = np.array([[-1,3,-2],[1,-1,-3],[3,-2,2],[1,2,1]])

#crea 1, 3 input
x_in = np.array([.5,.8,.2])

#crea un array con solo decimales de 7, 3
x_mat_in = np.array([[.5,.8,.2],[.1,.9,.6],[.2,.2,.3],
                     [.6,.1,.9],[.5,.5,.4],[.9,.1,.9],[.1,.8,.7]])

#define la funcion soft max
def soft_max_vec(vec):
    return np.exp(vec)/(np.sum(np.exp(vec)))

#define la funcion soft max mat
def soft_max_mat(mat):
    return np.exp(mat)/(np.sum(np.exp(mat),axis=1).reshape(-1,1))

print("------")

print('the matrix W_1\n pesos')
print(W_1)
print('-'*30)
print('vector input x_in\n input')
print(x_in)
print ('-'*30)
print('matrix input x_mat_in -- starts with the vector `x_in`\n')
print(x_mat_in)

print("------")

#antes de pasar por la primera capa
#Es la multiplicacion de dos matrices (producto escalar), resultado 1,4
z_2 = np.dot(x_in,W_1)
print(z_2)

print("------")

#calcula sigmoid en la primera capa
#Le ocupa sigmoid a los 4 valores, resultado 1,4
a_2 = sigmoid(z_2)
print(a_2)

print("------")

#Antes de pasar por la segunda capa
#Multiplicacion de matrices resultado 1,4
z_3 = np.dot(a_2,W_2)
print(z_3)

print("------")

#Aplica la activacion sigmoid en la segunda capa
#Aplica la activacion sigmoid a todos resultado 1,4
a_3 = sigmoid(z_3)
print(a_3)

print("------")

#Mutiplicacion pesos antes de pasar por el output, resultado 1,3
#dado que el ultimo peso tiene 3 columnas solamente
z_4 = np.dot(a_3,W_3)
print(z_4)

print("------")

#En la salida a todos los datos le aplica un soft max, resutlado 1,3
y_out = soft_max_vec(z_4)
print(y_out)

print("------")

#Esto hace lo mismo que lo anterior de forma mas reducida
def nn_comp_vec(x):
    return soft_max_vec(sigmoid(sigmoid(np.dot(x,W_1)).dot(W_2)).dot(W_3))

#Hace lo mismo que lo anterior, pero prueba con una entrada de 7, 3, por lo que la salida es 7,3
def nn_comp_mat(x):
    return soft_max_mat(sigmoid(sigmoid(np.dot(x,W_1)).dot(W_2)).dot(W_3))

print("------")

print(nn_comp_vec(x_in))

print("------")

print(nn_comp_mat(x_mat_in))


### Ejemplo 2

import pandas as pd
digits = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module1/L1/data/digits.csv")

#las predicciones de que numero es la imagen
labels = digits['label']

#los valores de cada pixel
digits = np.array(digits.drop('label', axis=1)).astype('float')
print(digits.shape, labels.shape)

#MUestra 5 imagenes al azar
import random
plt.figure(figsize=(12,4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(random.choice(digits).reshape(28,28))
    plt.axis("off")

split = 0.7, 0.3 # train, test
# normalize data
digits /= 255.0

split_ind = int(len(digits)*split[0])
X_train, X_test, y_train, y_test = digits[:split_ind], digits[split_ind:], labels[:split_ind], labels[split_ind:]
print(X_train.shape, X_test.shape)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'hidden_layer_sizes':[75, 100, 200],
              'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 
              'max_iter': [100, 200, 500, 800], 
              'learning_rate_init':[0.001, 0.01, 0.1, 1]}

model = MLPClassifier()
clf = GridSearchCV(estimator=model, param_grid=parameters, cv=5, n_jobs=-1)
clf.fit(X_train[:3000], y_train[:3000]) # may need to reduce the train set size to shorten the training time

print("The best parameter values found are:\n")
print(clf.best_params_)

# store the best model found in "bestmodel"
bestmodel = clf.best_estimator_

#Aca automaticamente despues hace un modelo con los mejores hiperparametros
from sklearn.metrics import accuracy_score

y_pred = bestmodel.predict(X_test)
print(f"The accuracy score of the best model is {accuracy_score(y_test, y_pred)}\n")

plt.figure(figsize=(12,8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    sample = random.choice(X_test)
    plt.imshow(sample.reshape(28,28))
    pred = bestmodel.predict(sample.reshape(1,-1))
    plt.title(f"Predicted as {pred}")
    plt.axis("off")

plt.tight_layout()
plt.show()


#######################################################################################################################

                                              #Gradiente Descendiente

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1234)  ## This ensures we get the same data if all of the other parameters remain fixed

num_obs = 100
#genera 100 valores distribuidos uniformemente, pueden aparecer valores entre 0 y 10
x1 = np.random.uniform(0,10,num_obs) #1,100
x2 = np.random.uniform(0,10,num_obs)


#genera una matriz cuadrada de 10 * 10 unos
const = np.ones(num_obs) #1, 100

#genera 100 numeros, entre con promedio 0 y std=0.5
eps = np.random.normal(0,.5,num_obs) #1, 100

b = 1.5
theta_1 = 2
theta_2 = 5

#multilicando todo y sumandolos todos
y = b*const+ theta_1*x1 + theta_2*x2 + eps

#crea una matriz, cada fila antrior se junta 3,100, pero con la transpuesta 100,3
x_mat = np.array([const,x1,x2]).T

#Hacemos un modelo de regresion lineal
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(fit_intercept=False)
lr_model.fit(x_mat, y)

#Sacamos todos los coeficientes del modelo, da el mismo resultado de la siguiente 1, 3
print(lr_model.coef_)

#Hace una multiplicacion de matrices, es una red neuronal con multiplicacion, resultado 1, 3
print(np.linalg.inv(np.dot(x_mat.T,x_mat)).dot(x_mat.T).dot(y))

learning_rate = 1e-3
num_iter = 10000
theta_initial = np.array([3,3,3])

#Gradiente descendiente de forma analitica, no siempre se puede hacer esto debido a que no es posible llegar
#a una solucion, pero si se puede converger para que llegue a algo
def gradient_descent(learning_rate, num_iter, theta_initial):

    ## Initialization steps
    theta = theta_initial
    theta_path = np.zeros((num_iter+1,3))
    theta_path[0,:]= theta_initial
    loss_vec = np.zeros(num_iter)

    ## Main Gradient Descent loop (for a fixed number of iterations)
    for i in range(num_iter):
        y_pred = np.dot(theta.T,x_mat.T)
        loss_vec[i] = np.sum((y-y_pred)**2)
        grad_vec = (y-y_pred).dot(x_mat)/num_obs  #sum up the gradients across all observations and divide by num_obs
        grad_vec = grad_vec
        theta = theta + learning_rate*grad_vec
        theta_path[i+1,:]=theta
    return theta_path, loss_vec

true_coef = [b, theta_1, theta_2]

def plot_ij(theta_path, i, j, ax):
    ax.plot(true_coef[i], true_coef[j],
            marker='p', markersize=15, label='true coef', 
            color='#778899')
    ax.plot(theta_path[:, i],theta_path[:, j],
            color='k', linestyle='--', marker='^', 
            markersize=5, markevery=50)
    ax.plot(theta_path[0, i], theta_path[0, j], marker='d', 
            markersize=15, label='start', color='#F08080')
    ax.plot(theta_path[-1, i], theta_path[-1, j], marker='o', 
            markersize=15, label='finish', color='#F08080')
    ax.set(
        xlabel='theta'+str(i),
        ylabel='theta'+str(j))
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc='best')

def plot_all(theta_path, loss_vec, learning_rate, num_iter, theta_initial, gdtype='Gradient Descent'):
    fig = plt.figure(figsize=(16, 16))
    title = '{gdtype} in the 3d parameter space - Learning rate is {lr} // {iters} iters // starting point {initial}'
    title = title.format(gdtype=gdtype, lr=learning_rate, 
                         iters=num_iter, initial=theta_initial)
    fig.suptitle(title, fontsize=20)
    ax = fig.add_subplot(2, 2, 1)
    plot_ij(theta_path, 0, 1, ax)
    ax = fig.add_subplot(2, 2, 2)
    plot_ij(theta_path, 0, 2, ax)
    ax = fig.add_subplot(2, 2, 3)
    plot_ij(theta_path, 1, 2, ax)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(loss_vec)
    ax.set(xlabel='iterations', ylabel='squared loss')
    ax.grid(True)
    plt.show()

theta_path, loss_vec = gradient_descent(learning_rate, num_iter, theta_initial)
plot_all(theta_path, loss_vec, learning_rate, num_iter, theta_initial)


#En la parte anterior saca un promedio de gradiente con todo el conjunto de datos

#Ahora toma en cuanta todos los datos y hace que tengan "reaccion exagerada", pero deberan promediarse

def stochastic_gradient_descent(learning_rate, num_iter, theta_initial):

    ## Initialization steps
    theta = theta_initial
    # below are different in STOCHASTIC gradient descent
    theta_path = np.zeros(((num_iter*num_obs)+1,3))
    theta_path[0,:] = theta_initial
    loss_vec = np.zeros(num_iter*num_obs)

    ## Main SGD loop
    count = 0
    for i in range(num_iter):
        for j in range(num_obs):
            count+=1
            y_pred = np.dot(theta.T,x_mat.T)
            loss_vec[count-1] = np.sum((y-y_pred)**2)
            grad_vec = (y[j]-y_pred[j])*(x_mat[j,:])
            theta = theta + learning_rate*grad_vec
            theta_path[count,:]=theta
    return theta_path, loss_vec

## Parameters to play with
learning_rate = 1e-4
num_iter = 100
theta_initial = np.array([3, 3, 3])


theta_path, loss_vec = stochastic_gradient_descent(learning_rate, 
                                                   num_iter, 
                                                   theta_initial)
plot_all(theta_path, loss_vec, learning_rate, 
         num_iter, theta_initial, 'SGD')


#Probando otros parametros
learning_rate = 1e-4
num_iter = 100
theta_initial = np.array([3,3,3])


theta_path, loss_vec = stochastic_gradient_descent(learning_rate, 
                                                   num_iter, 
                                                   theta_initial)
plot_all(theta_path, loss_vec, learning_rate, num_iter, 
         theta_initial, 'SGD')


#######################################################################################################################

                                                #Regression con Keras

import warnings
warnings.simplefilter('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
from itertools import accumulate
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
from sklearn.metrics import mean_squared_error

sns.set_context('notebook')
sns.set_style('white')

raw_dataset = pd.read_csv('day.csv')

print(raw_dataset.sample(5))

#Se eliminan las columnas clasificatorias
raw_dataset = raw_dataset.drop(columns=['dteday', 'instant', 'registered', 'casual'])

#Graficamos solo 4 columnas contra "cnt", regresiones lineales en los datos
col = ['temp', 'atemp', 'hum', 'windspeed']
plt.figure(figsize=[20,12])
for i in enumerate(col):
    plt.subplot(2,2,i[0]+1)
    sns.regplot(data=raw_dataset,x=i[1],y='cnt',line_kws={"color":'red'})
    
plt.show()

#Se muestran graficos de cajas de estas columnas contra "cnt"
col = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
plt.figure(figsize=[20,12])
for i in enumerate(col):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(data=raw_dataset,x=i[1],y='cnt')
    
plt.show()

#Hacemos un pairplot de toda la data
#sns.pairplot(raw_dataset)
#plt.show()

#Tambien las correlaciones y su matriz, muestran una correlacion bastante baja en todas las columnas
plt.figure(figsize=[15,8])
fig = sns.heatmap(raw_dataset.corr(),cmap='hot_r',
            annot=True,linecolor='black',linewidths=0.01,annot_kws={"fontsize":12},fmt="0.2f")

top, bottom = fig.get_ylim()
fig.set_ylim(top+0.1,bottom-0.1)

left, right = fig.get_xlim()
fig.set_xlim(left-0.1,right+0.1) 

plt.yticks(fontsize=13,rotation=0)
plt.xticks(fontsize=13,rotation=90)
plt.show()

#Separamos la data en data de entrenaiento y prueba
train_dataset = raw_dataset.sample(frac=0.8, random_state=0)
test_dataset = raw_dataset.drop(train_dataset.index)

#copiamos las data
train_features = train_dataset.copy()
test_features = test_dataset.copy()

#Elimina todas la data solo deja "cnt" sin nombre
train_labels = train_features.pop('cnt')
test_labels = test_features.pop('cnt')


#La normalizacion a los datos se debe hacer debido a que los datos contiene diferentes escalas y al aplicarle, pesos
#no se entenera que se le aplican a todos por igual, por lo que se deben hacer, ademas el proceso es mucho mas estable
#en una red neuronal

#Crea una capa que normaliza las entradas que ingresen a esa capa, con media=0 y std=1
#-1 significa que normaliza por el indice o cantidad de datos
normalizer = tf.keras.layers.Normalization(axis=-1)

#Aca es donde ajustamos la normalizacion con estos datos para tomar una media y una varianza
normalizer.adapt(np.array(train_features))

#Entrega una media para cada columnas
print(normalizer.mean.numpy())

#Entrega una varianza para cada columna
print(normalizer.variance.numpy())

#Muestra solo la primera fila
print(np.array(train_features[:1]))

#ACA APLICA LA NORMALIZACION
#Lo que hace es tener 11 medias y 11 varianzas, para columnas, son las ateriores las que son realmente
#reconoce que es una fila con 11 columnas
#aplica esta formula dependiendo de en que columna esta el datos, son 11 datos justos
#(input-mean)/sqrt(var)
#tomara diferentes medias y varianzas segun la columna que este el input
print(normalizer(np.array(train_features[:1])).numpy())

print(train_features.head())

#Solo tomamos una columnas porque es alta mente correlacionada, hara buenas predicciones con "cnt"
temp = np.array(train_features['temp'])

#None en axis es igual a poner -1, solo una columnas para normazliacion
temp_normalizer = layers.Normalization(input_shape=[1,], axis=None)

#Tome media y varianza de temp
temp_normalizer.adapt(temp)

#Hacemos una secuencia, tambien llamada arquitectura de modelo
#primero una normalizacion, luego una capa oculta con layers.Dense(1)
temp_model = tf.keras.Sequential([temp_normalizer,layers.Dense(units=1)
])
print(temp_model.summary())


#compile es para configurar un modelo ya creado, le decimos cual es el optimizador gradiente descendiente estocastico
#funcion de perdida error cuadratico medio
#el optimicador, optmiza los pesos del modelo y la taza de aprendizaje
#la funcion de perdida minimiza el error
temp_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='mean_squared_error')


#Ajustamos el modelo, con la columna de entrenamiento "temp", luego el label que es "cnt"
history = temp_model.fit(
    train_features['temp'],
    train_labels,
    # to view the logs, uncomment this:
    verbose=False,
    epochs=100,
    # validation split: 20% of the training data.
    validation_split = 0.2)

#definimos la funcion de perdida
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.grid(True)
    plt.show()

#con el modelo ajustado anterior mostramos la funcion de perdida
plot_loss(history)

#hacemos predicciones con el modelo
y_pred= temp_model.predict(test_features['temp'])

#Muestra el error de la prediccion y los datos de prueba
print(np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))

#Muestra los puntos y la regression del modelo
plt.figure(figsize = (4,4))
plt.plot(test_features['temp'], test_labels,'o',
         test_features['temp'],temp_model.predict(test_features['temp']),'g')
plt.show()


###Ejemplo 2

#El anterior fue solo una columnas input ahora con multiples columnas input

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='mean_squared_error')


history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # to view the logs, uncomment this:
    verbose=0,
    # validation split: 20% of the training data.
    validation_split = 0.2)

plot_loss(history)

y_pred= linear_model.predict(test_features)

#Error del modelo
print(np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))

#Muestra los nombres de las columnas
print("Column names")
print(test_features.columns)


#En la segunda iteracion se ve la capa real
for layer in linear_model.layers:
    #Son dos capas primero una capa solo para mostrar la normalizacion
    #luego la segunda es la real, porque es solo 1, en la segunda muestra las ponderaciones
    #esto nos dice que tan importante es para el modelo
    #donde la segunda, la octaba y la novena son las que mas aportan al modelo o predicciones
    #"yt", "temp", "atemp"
    print("Layer Name:", layer.name)
    print("---")
    print("Weights")
    #Son 11 pesos 1 por columnas (11,1), aparece la ponderacion
    print("Shape:",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
    print("---")
    #Lo bias es solo 1, pero en la normalizacion se le cambia el valor segun la columna
    print("Bias")
    print("Shape:",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')

###Ejemplo 3

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module1/L2/data/IowaHousingPrices.csv')

x_train = df[['SquareFeet']].values[1:-500]
y_train = df[['SalePrice']].values[1:-500]
x_test = df[['SquareFeet']].values[-500:]
y_test = df[['SalePrice']].values[-500:]

output_size=1
hidden_layer=500
input_size=1
learning_rate=0.51
loss_function='mean_squared_error'
epochs=30
batch_size=10

#Creamos la arquitectura del modelo
#dos capas una oculta y otra de salida, con activacion relu
model = keras.Sequential()

#model.add es para añadir capas, debe ser antes de compilar y fitear
model.add(keras.layers.Dense(hidden_layer,  activation='relu'))
model.add(keras.layers.Dense(output_size))

#Para configurar el modelo con la funcion de perdida y el optimizador
model.compile(keras.optimizers.Adam(learning_rate=learning_rate), loss_function)

#Ajustamos el modelo
model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

print(x_test.shape)

#Crea un array es una fila, valor de inicia, maximo, saltos de 10, luego lo transforma a columna
#crea una nueva data x
x=np.arange(x_test.min(),x_test.max(),10).reshape(-1,1)

print(x.shape)

#Crea una nueva data esto y
y_pred = model.predict(x)

#for layer in model.layers:
    #Son dos capas primero una capa solo para mostrar la normalizacion
    #luego la segunda es la real, porque es solo 1, en la segunda muestra las ponderaciones
    #esto nos dice que tan importante es para el modelo
    #donde la segunda, la octaba y la novena son las que mas aportan al modelo o predicciones
    #"yt", "temp", "atemp"
#    print("Layer Name:", layer.name)
#    print("---")
#    print("Weights")
    #Son 11 pesos 1 por columnas (11,1), aparece la ponderacion
#    print("Shape:",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
#    print("---")
    #Lo bias es solo 1, pero en la normalizacion se le cambia el valor segun la columna
#    print("Bias")
#    print("Shape:",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')

#Muestra los puntos de prueba que solo los utilizo para crear otra data
#Tambien muestra un regreson con la predicciones
plt.plot(x,y_pred,label="prediction ")
plt.plot(x_test,y_test,'ro',label="test samples")
plt.xlabel('Input')
plt.ylabel('Predicted Output')
plt.legend()
plt.show()


###Ejemplo 4

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module1/L2/data/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.dropna()
dataset = dataset.drop(columns=['Origin'])

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
print(normalizer.variance.numpy())

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
linear_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='mean_squared_error')

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # to view the logs, uncomment this:
    verbose=0,
    # validation split: 20% of the training data.
    validation_split = 0.2)

plot_loss(history)

y_pred= linear_model.predict(test_features)
print(np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))


#######################################################################################################################

                                          #Cargando Imagenes en Keras

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tensorflow INFO and WARNING messages are not printed 

import random 

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import PIL
import PIL.Image
from PIL import Image, ImageOps
import tensorflow as tf

import keras
from keras.preprocessing import image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img 
from keras.models import Model
from keras.layers import Input, Dense

import cv2

sns.set_context('notebook')
sns.set_style('white')

#Visualiza imagenes
def visualize(X_train):
    plt.rcParams['figure.figsize'] = (6,6) 

    for i in range(4):
        plt.subplot(2,2,i+1)
        num = random.randint(0, len(X_train))
        plt.imshow(X_train[num], cmap='gray', interpolation='none', vmin=0, vmax=255)
        plt.title("class {}".format(y_train[num]))
    
    plt.tight_layout()
    plt.show()
print(tf.__version__)

#Este metodo los descarga y los usa en el mismo programa, solo disponible por keras
#PERO SOLO LOS DESCARGA LA PRIMERA VEZ

#descargamos el datasets, mnist y tenemos listo las diviciones, datos de entrenamiento, etc
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(X_train.shape)
print(X_train.dtype)
print(y_train.shape)
print(y_train.dtype)

#mnist muestra 4 imagenes de numeros
visualize(X_train)

ENCODING_DIM = 64

# Encoded representations:
inputs = Input(shape=(784,)) 
encoded = Dense(ENCODING_DIM, activation="sigmoid")(inputs)

# Reconstructions:
encoded_inputs = Input(shape=(ENCODING_DIM,), name='encoding')
reconstruction = Dense(784, activation="sigmoid")(encoded_inputs)

print("Encoded Input: ", encoded.shape)
print("Reconstructed Input: ", reconstruction.shape)

#Esta es otra data importada
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

#fashion mnist muestra imagenes de ropa
visualize(X_train) 

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

print(X_train.shape)
print(X_train.dtype)

#cifar tiene imagenes de cualquier cosa
visualize(X_train)

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data(label_mode = 'fine')

print(X_train.shape)
print(X_train.dtype)

#cifar100 tambien tiene imagenes de cualquier cosa
visualize(X_train)

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data(label_mode = 'coarse')

#muestra diferentes imagenes
visualize(X_train)

###opcion de descarga 2

#Desde una pagina web

import requests

#Importa la imagen de un perro solo esa
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module3/L1/Dog_Breeds.jpg'
filename = URL.split('/')[-1]
r = requests.get(URL, allow_redirects=True)
open(filename, 'wb').write(r.content)

img_height, img_width = 100, 100

print("PIL1")
gray_img = load_img(
    'Dog_Breeds.jpg',
    target_size=(img_height, img_width),
    interpolation='nearest', color_mode = "grayscale")
    
print("PIL2")
color_img = load_img(
    'Dog_Breeds.jpg',
    target_size=(img_height, img_width),
    interpolation='nearest')


print(type(gray_img))
print(type(color_img))

plt.imshow(gray_img,cmap="gray")
plt.show()

plt.imshow(color_img)
plt.show()

#convertimos la imagen a un array 3D

#Keras puede transformar una imagen a array automaticaente
input_arr = tf.keras.preprocessing.image.img_to_array(color_img)

#ahora tiene dimension 100, 100, 3 que es 100*100*3
print("image shape",input_arr.shape)

#Necesitamos agregar la direccion al lote antes de usar en keras, le agrega un dato mas 1, 100, 100, 3
#puede ser cualquierda de las siguientes formas
input_arr_batch = np.array([input_arr])

input_arr_batch=input_arr.reshape(-1,input_arr.shape[0],input_arr.shape[1],input_arr.shape[2])

print("image shape plus batch dimension",input_arr_batch.shape)

color_img = tf.keras.preprocessing.image.array_to_img(input_arr)
plt.imshow(color_img)
plt.show()

#Para guardar una imagen con keras
tf.keras.preprocessing.image.save_img('dog_color_img.jpg', color_img)


###Ejemplo 4 descarga en un directorio, lo puedes ocupar sin el .txt, etc

import pathlib
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/datasets/flower_photos.tgz"

# Download the data and track where it's saved using tf.keras.utils.get_file in a variable called data_dir
data_dir = keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

for folder in data_dir.glob('[!LICENSE]*'):
    print('The', folder.name, 'folder has', len(list(folder.glob('*.jpg'))), 'pictures')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count, 'total images')

dandelion = list(data_dir.glob('dandelion/*'))
PIL.Image.open(str(dandelion[1]))

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))

sunflowers = list(data_dir.glob('sunflowers/*'))
PIL.Image.open(str(sunflowers[1]))

daisy = list(data_dir.glob('daisy/*'))
PIL.Image.open(str(daisy[1]))

#La cantidad de imagenes
batch_size = 32

# Here we set the size of all the images to be 200x200
img_height = 200
img_width = 200

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

validation_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# .take() will take the first batch from a tensorflow dataset. 
# In our case it has taken the first 32 images.
first_batch = train_ds.take(1)

plt.figure(figsize = (25,10))
for img, lbl in first_batch:
    # lets look at the first 10 images
    for i in np.arange(10):
        plt.subplot(2,5,i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(class_names[lbl[i]])
        plt.axis("off")
plt.show()

import tensorflow as tf

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    vertical_flip=True,
)

flowers_data = img_gen.flow_from_directory(data_dir)
images, labels = next(flowers_data)

print(images.shape)
print(labels.shape)

plt.figure(figsize=(25, 10))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(images[i])
    index = [index for index, each_item in enumerate(labels[i]) if each_item]
    plt.title(list(flowers_data.class_indices.keys())[index[0]])
    plt.axis("off")
plt.show()


###Ocupar desde URL

def load_image(link, target_size=None):
    import requests
    import shutil
    import os
    _, ext = os.path.splitext(link)
    r = requests.get(link, stream=True)
    with open('temp.' + ext, 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)
    img = tf.keras.preprocessing.image.load_img('temp.' + ext, target_size=target_size)
    return tf.keras.preprocessing.image.img_to_array(img)

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module3/L1/Dog_Breeds.jpg'

img = load_image(URL, target_size=(100, 100))
plt.imshow(img/255)
plt.show()


#######################################################################################################################

                                         #Ejemplo Redes Neuronales

import warnings
import skillsnetwork

warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tensorflow INFO and WARNING messages are not printed 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from keras.models  import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop

diabetes_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module2/L2/diabetes.csv", names=["times_pregnant", "glucose_tolerance_test", "blood_pressure", "skin_thickness", "insulin", "bmi", "pedigree_function", "age", "has_diabetes"], header=0)

print(diabetes_df.shape)
print(diabetes_df.sample(5))

X = diabetes_df.iloc[:, :-1].values
y = diabetes_df["has_diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11111)

print(np.mean(y), np.mean(1-y))

#Sin haber hecho un modelo, con los datos disponible vamos a tener un porcentaje predictivo de 65% porque los porcentajes
#de datos con diabetes es 65% por lo tanto, si queremos saber quien tiene diabetes tenemos 65%

#Baseline con random forest

#Veamos un modelo de randomforestclassifer
rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X_train, y_train)

y_pred_class_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)


print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_rf)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_rf[:,1])))

#ploteamos roc
def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title='ROC Curve for {} on PIMA diabetes problem'.format(model_name),
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    plt.show()    
plot_roc(y_test, y_pred_prob_rf[:, 1], 'RF')

#tiene una exactitud de 0.76

#escalamos los datos
normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

#creamos un modelo secuencial
model_1 = Sequential()

#una capa de 12 nodos, activacion sigmoid, input_shape es informacion de cuantos datos de entrada seran
#no es necesario poner input_shape solo si lo quiere especificar
model_1.add(Dense(12,input_shape = (8,),activation = 'sigmoid'))
#una capa de salida con activacion sigmoid
model_1.add(Dense(1,activation='sigmoid'))

print(model_1.summary())

model_1.compile(SGD(learning_rate = .003), "binary_crossentropy", metrics=["accuracy"])

#576, 8 para X_train_norm
#576, 1 para y_train
run_hist_1 = model_1.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=200)

y_pred_class_nn_1 = np.argmax(model_1.predict(X_test_norm), axis=-1)
y_pred_prob_nn_1 = model_1.predict(X_test_norm)

#predijo solo 0
print(y_pred_class_nn_1[:10])

#predijo valores cercanos a 0.5
print(y_pred_prob_nn_1[:10])

#exactitu baja de 0.65, debido a que solo pronostico un tipo de clase
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_1)))

#con probabilidad es mejor de 0.77
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_1)))

plot_roc(y_test, y_pred_prob_nn_1, 'NN')

#Es para mostrar las metricas del modelo ajustado
#key se ocupa para obtener los nombres de un diccionario (el nombre de la columna)
run_hist_1.history.keys()

#Funcion de perdida del modelo
fig, ax = plt.subplots()
ax.plot(run_hist_1.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()
plt.show()

#AJustamos otro modelo con 1000 iteraciones para ver si mejora el error
run_hist_1b = model_1.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=1000)

#Obtiene la cantidad que tiene la funcion de perdida
n = len(run_hist_1.history["loss"])
m = len(run_hist_1b.history['loss'])
fig, ax = plt.subplots(figsize=(16, 8))

#ploteo de la funcion de perdida
ax.plot(range(n), run_hist_1.history["loss"],'r', marker='.', label="Train Loss - Run 1")
ax.plot(range(n, n+m), run_hist_1b.history["loss"], 'hotpink', marker='.', label="Train Loss - Run 2")

ax.plot(range(n), run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss - Run 1")
ax.plot(range(n, n+m), run_hist_1b.history["val_loss"], 'LightSkyBlue', marker='.',  label="Validation Loss - Run 2")

ax.legend()
plt.show()

#no se beneficio con mas iteraciones el modelo


###Ejemplo 2

#Creamos otro modelo
model_2 = Sequential()
model_2.add(Dense(6, input_shape=(8,), activation="relu"))
model_2.add(Dense(6,  activation="relu"))
model_2.add(Dense(1, activation="sigmoid"))

model_2.compile(SGD(learning_rate = .003), "binary_crossentropy", metrics=["accuracy"])

#Ocupamos mucho mas iteraciones para ver si es realmente mejor el modelo
run_hist_2 = model_2.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=1500)

print(run_hist_2.history.keys())

n = len(run_hist_2.history["loss"])

#PLotemoas las funcion de perdida
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
ax.plot(range(n), (run_hist_2.history["loss"]),'r.', label="Train Loss")
ax.plot(range(n), (run_hist_2.history["val_loss"]),'b.', label="Validation Loss")
ax.legend()
ax.set_title('Loss over iterations')
plt.show()

ax = fig.add_subplot(1, 2, 2)
ax.plot(range(n), (run_hist_2.history["acc"]),'r.', label="Train Acc")
ax.plot(range(n), (run_hist_2.history["val_acc"]),'b.', label="Validation Acc")
ax.legend(loc='lower right')
ax.set_title('Accuracy over iterations')
plt.show()

y_pred_class_nn_2 = model_2.predict_classes(X_test_norm)
y_pred_prob_nn_2 = model_2.predict(X_test_norm)
print('')
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_2)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_2)))

plot_roc(y_test, y_pred_prob_nn_2, 'NN-2')

#No mejoro con las iteraciones, los errores tienden a cierto valor minimo y se queda estables



#######################################################################################################################

                                              #Retropropagacion

import numpy as np
import matplotlib.pyplot as plt

## This code below generates two x values and a y value according to different patterns
## It also creates a "bias" term (a vector of 1s)
## The goal is then to learn the mapping from x to y using a neural network via back-propagation

#Crea los pesos
num_obs = 500
x_mat_1 = np.random.uniform(-1,1,size = (num_obs,2))
x_mat_bias = np.ones((num_obs,1))
x_mat_full = np.concatenate( (x_mat_1,x_mat_bias), axis=1)

# PICK ONE PATTERN BELOW and comment out the rest.

# # Circle pattern
# y = (np.sqrt(x_mat_full[:,0]**2 + x_mat_full[:,1]**2)<.75).astype(int)

#Crea una funcion para que los datos esten en forma de diamante
y = ((np.abs(x_mat_full[:,0]) + np.abs(x_mat_full[:,1]))<1).astype(int)

# # Centered square
# y = ((np.maximum(np.abs(x_mat_full[:,0]), np.abs(x_mat_full[:,1])))<.5).astype(int)

# # Thick Right Angle pattern
# y = (((np.maximum((x_mat_full[:,0]), (x_mat_full[:,1])))<.5) & ((np.maximum((x_mat_full[:,0]), (x_mat_full[:,1])))>-.5)).astype(int)

# # Thin right angle pattern
# y = (((np.maximum((x_mat_full[:,0]), (x_mat_full[:,1])))<.5) & ((np.maximum((x_mat_full[:,0]), (x_mat_full[:,1])))>0)).astype(int)

#500,3 y 500,1
print('shape of x_mat_full is {}'.format(x_mat_full.shape))
print('shape of y is {}'.format(y.shape))

#Crea un plot con los datos como diamantes
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x_mat_full[y==1, 0],x_mat_full[y==1, 1], 'ro', label='class 1', color='darkslateblue')
ax.plot(x_mat_full[y==0, 0],x_mat_full[y==0, 1], 'bx', label='class 0', color='chocolate')
# ax.grid(True)
ax.legend(loc='best')
ax.axis('equal')
plt.show()

def sigmoid(x):
    
    #Sigmoid function
 
    return 1.0 / (1.0 + np.exp(-x))


def loss_fn(y_true, y_pred, eps=1e-16):
    
    #Loss function we would like to optimize (minimize)
    #We are using Logarithmic Loss
    #http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
    
    y_pred = np.maximum(y_pred,eps)
    y_pred = np.minimum(y_pred,(1-eps))
    return -(np.sum(y_true * np.log(y_pred)) + np.sum((1-y_true)*np.log(1-y_pred)))/len(y_true)


def forward_pass(W1, W2):
    
    #Does a forward computation of the neural network
    #Takes the input `x_mat` (global variable) and produces the output `y_pred`
    #Also produces the gradient of the log loss function
    
    global x_mat
    global y
    global num_
    #El modelo lo hace artesanalmente
    z_2 = np.dot(x_mat, W_1)
    a_2 = sigmoid(z_2)
    z_3 = np.dot(a_2, W_2)
    y_pred = sigmoid(z_3).reshape((len(x_mat),))
    # Ahora hace el gradiente
    J_z_3_grad = -y + y_pred
    J_W_2_grad = np.dot(J_z_3_grad, a_2)
    a_2_z_2_grad = sigmoid(z_2)*(1-sigmoid(z_2))
    J_W_1_grad = (np.dot((J_z_3_grad).reshape(-1,1), W_2.reshape(-1,1).T)*a_2_z_2_grad).T.dot(x_mat).T
    gradient = (J_W_1_grad, J_W_2_grad)
    
    # return
    return y_pred, gradient


def plot_loss_accuracy(loss_vals, accuracies):
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Log Loss and Accuracy over iterations')
    
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss_vals)
    ax.grid(True)
    ax.set(xlabel='iterations', title='Log Loss')
    
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracies)
    ax.grid(True)
    ax.set(xlabel='iterations', title='Accuracy')
    plt.show()


#### Initialize the network parameters

np.random.seed(1241)

W_1 = np.random.uniform(-1,1,size=(3,4))
W_2 = np.random.uniform(-1,1,size=(4))
num_iter = 5000
learning_rate = .001
x_mat = x_mat_full


loss_vals, accuracies = [], []
for i in range(num_iter):
    ###Calcula el modelo y el gradiente artesanalmente
    y_pred, (J_W_1_grad, J_W_2_grad) = forward_pass(W_1, W_2)
    
    ##Actualiza las matrices de pesos en cada itineracion
    W_1 = W_1 - learning_rate*J_W_1_grad 
    W_2 = W_2 - learning_rate*J_W_2_grad
    
    ### Compute the loss and accuracy
    curr_loss = loss_fn(y,y_pred)
    loss_vals.append(curr_loss)
    acc = np.sum((y_pred>=.5) == y)/num_obs
    accuracies.append(acc)

    ## Muestra como cambia el error y la exactitud en cada iteracion
    if((i%200) == 0):
        print('iteration {}, log loss is {:.4f}, accuracy is {}'.format(
            i, curr_loss, acc
        ))

#Muestra la funcion de perdida y la funcion de la precision como varia con las iteraciones
#vemos que el error tienede a cierto valor y se queda estatico
#mientras que la precision mejora en cada iteracion hasta que se acerca mucho a uno
plot_loss_accuracy(loss_vals, accuracies)


pred1 = (y_pred>=.5)
pred0 = (y_pred<.5)

fig, ax = plt.subplots(figsize=(8, 8))
# true predictions
ax.plot(x_mat[pred1 & (y==1),0],x_mat[pred1 & (y==1),1], 'ro', label='true positives')
ax.plot(x_mat[pred0 & (y==0),0],x_mat[pred0 & (y==0),1], 'bx', label='true negatives')
# false predictions
ax.plot(x_mat[pred1 & (y==0),0],x_mat[pred1 & (y==0),1], 'yx', label='false positives', markersize=15)
ax.plot(x_mat[pred0 & (y==1),0],x_mat[pred0 & (y==1),1], 'yo', label='false negatives', markersize=15, alpha=.6)
ax.set(title='Truth vs Prediction')
ax.legend(bbox_to_anchor=(1, 0.8), fancybox=True, shadow=True, fontsize='x-large')
plt.show()



#######################################################################################################################

                                   #Optimizadores en gradiente descendiente

#El fin de ocupar optimizadores es que el modelo mejore, por lo que en este codigo veremos como podemos mejorarlos

import pandas as pd
import numpy as np
import time
import sys

import warnings
warnings.simplefilter('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, RMSprop, Adam, Adagrad
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

#Es la data de nutricion, se recomienda cual comida se debe comer con mas frecuencia, menos frecuencia, con moderacion
#es un problema de clasificacion
food_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/datasets/food_items.csv")

print(food_df.shape)

print(food_df.dtypes)

# Get the row entries with the last col 'class'
print(food_df.iloc[:, -1:].value_counts(normalize=True))
print(food_df.iloc[:, -1:].value_counts().plot.bar(color=['#e67e22', '#27ae60', '#2980b9']))

X_raw = food_df.iloc[:, :-1]
y_raw = food_df.iloc[:, -1:]

# Escalamos los datos
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

#Codificamos los datos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())

rs = 123

#Separamos en datos de prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)
print(f"Training dataset shape, X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing dataset shape, X_test: {X_test.shape}, y_test: {y_test.shape}")

#Hacemos un modelo de redes neuronales con Sklearn
base_ann = MLPClassifier(random_state=rs,  hidden_layer_sizes=(32, 8), 
                    solver='sgd', momentum=0, 
                    early_stopping=True,
                    max_iter=100)

#Para mostrar el tiempo de ajuste
def fit_and_score(model, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start
    n_iter = model.n_iter_
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    loss_curve = model.loss_curve_
    return round(fit_time, 2), n_iter, train_score, test_score

#Lo ajustamos y guardamos su puntuacion
fit_time, n_iter, train_score, test_score = fit_and_score(base_ann, X_train, X_test, y_train, y_test)

#Obtiene un resultado bastante malo con 12 iteraciones, exactitud=0.42 en datos de entrenamiento como en datos
#de prueba
print(f"Training converged after {n_iter} iterations with train score (accuracy) {round(train_score, 2)} \
and test score (accuracy) {round(test_score, 2)}.")

#IMPORTANTE!!
#Los modelos de redes neuronales tienen funcionamiento muy complejos por lo que necesitan una gran configuracion
#debido a esto, en este caso al poner un modelo muy simple y/o pocas iteraciones o falta de datos de entrenamiento,
#el modelo simplemente no funciono

#Una razón común es que el SGD quedó atrapado en uno de los mínimos locales!!!
#Si la función de costo tiene múltiples mínimos o incluso puntos de silla (donde los gradientes son muy pequeños
#o ceros, por lo que las ponderaciones no se actualizarán), el SGD puede quedarse estancado en los mínimos locales.


#Definimos un modelo de regression logisitca
lr_model = LogisticRegression(random_state=rs, max_iter = 200)
lr_model.fit(X_train, y_train)
lr_score = lr_model.score(X_test, y_test)
print(f"The test score for the logistic regression is {round(lr_score, 2)}")

#Este modelo mucho mas simple logro mejor puntuacion de 0.77

def draw_cost(X, Y, Z, title):
    fig = plt.figure()
    fig.set_size_inches(8, 6, forward=True)
    fig.set_dpi(100)
    ax = plt.axes(projection='3d')
    ax.view_init(30, 35)
    ax.contour3D(X, Y, Z, 100, cmap=plt.cm.coolwarm)
    ax.set_title(title)

def one_mini_function():
    w1 = np.linspace(-10, 10, 50)
    w2 = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(w1, w2)
    Z = np.log(np.sqrt(X ** 2 + Y ** 2))
    return X, Y, Z

X, Y, Z = one_mini_function()
draw_cost(X, Y, Z, "Cost function with the global minima")

def multi_mini_function():
    w1 = np.linspace(-10, 11, 50)
    w2 = np.linspace(-10, 11, 50)
    X, Y = np.meshgrid(w1, w2)
    Z = X ** 2  - Y ** 2 + 4*X
    return X, Y, Z

X1, Y1, Z1 = multi_mini_function()
draw_cost(X1, Y1, Z1, "Cost function with a saddle point in the middle")

#Para resolver el estancamiento del gradiente y la funcion de costo, podemos añadir ruido
#Los pesos siempre tienden a moverse lentamente hacia el optimo

#Al añadir un gradiente este puede ir tomando promedios con el fin de que tome en cuenta los pasos que va dando
#Esto le da un impulso hacia abajo que se va reduciendo, pero va asegurar que el modelo llegue al optimo


#Creamos el modelo con gradiente "sgd"
momentum_ann = MLPClassifier(random_state=123,  hidden_layer_sizes=(32, 8), 
                    solver='sgd', momentum=0.9, 
                    early_stopping=True,
                    max_iter=100)
fit_time, n_iter, train_score, test_score = fit_and_score(momentum_ann, X_train, X_test, y_train, y_test)
print(f"Training converged after {n_iter} iterations with test score (accuracy) {round(test_score, 2)}")

#Mejoro la puntuacion bastante a 0.74 con 93 iteraciones


###Ahora con impulso nesterov

#Nesterov le da un impulso adicional al gradiente, esto evita que se quede estancado, pero tambien puede pasarse
#del minimo

#Modelo con gradiente y impulso nesterovs
nesterovs_ann = MLPClassifier(random_state=123,  hidden_layer_sizes=(32, 8), 
                    solver='sgd', momentum=0.95, 
                    nesterovs_momentum=True,
                    early_stopping=True,
                    max_iter=100)
fit_time, n_iter, train_score, test_score = fit_and_score(nesterovs_ann, X_train, X_test, y_train, y_test)
print(f"Training converged after {n_iter} iterations with score (accuracy) {round(test_score, 2)}")

#Mejoro super poco exactitud=0.74, pero lo hizo con menos iteraciones con 83


###Adam

#Es posible que queremaos una optmizacion adaptativa, es decir que cambie, que al principio pueda ser rapida y 
#despues lenta, para evitar sobrepasar el nivel optimo

#Modelo con adam
adam_ann = MLPClassifier(random_state=123,  hidden_layer_sizes=(32, 8), 
                    solver='adam',
                    early_stopping=True,
                    max_iter=100)
fit_time, n_iter, train_score, test_score = fit_and_score(adam_ann, X_train, X_test, y_train, y_test)
print(f"Training converged after {n_iter} iterations with score (accuracy) {round(test_score, 2)}")

#Mejoro bastante el modelo con una exactitud=0.84 en 73 iteraciones


###formulas de Adam

# cost function
def cost_function(w):
    return (w - 4) ** 2 + 2*w

## take derivative
def grad_function(w):
    return 2*(w-4) + 2

def plot_cost():
    fig, axis = plt.subplots()
    fig.set_size_inches(8, 6, forward=True)
    fig.set_dpi(100)

    x = np.linspace(0,6,100)
    y = cost_function(x)
    axis.plot(x, y, 'b')
    axis.set_xlabel("Weight")
    axis.set_ylabel("Cost")

plot_cost()

def is_converged(w0, w1):
    return abs(w0 - w1) <= 1e-6

# Implement Adam
def adam(t, w, dw, m, v, alpha = 0.1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    # para
    # First-moment
    m = beta1*m + (1-beta1)*dw
    # Second-moment
    v = beta2*v + (1-beta2)*(dw**2)
    # Bias correction
    m_unbiased = m/(1-beta1**t)
    v_unbiased = v/(1-beta2**t)
    # Update weights
    w = w - alpha*(m_unbiased/(np.sqrt(v_unbiased) + epsilon))
    return w, m, v

w,m,v,t = 0,0,0,1
converged = False
w_his = []
loss_his = []
while not converged:
    dw = grad_function(w)
    w_prev = w
    w, m, v = adam(t, w, dw, m, v)
    loss_his.append(cost_function(w))
    w_his.append(w)
    if is_converged(w, w_prev):
        break
    else:
        t+=1

def plot_cost_w_grad(w_his, loss_his):
    fig, axis = plt.subplots()
    fig.set_size_inches(8, 6, forward=True)
    fig.set_dpi(100)
    x = np.linspace(-2,6,100)
    y = cost_function(x)
    axis.scatter(w_his, loss_his, marker='<', color='r', s=18)
    axis.plot(x, y, 'grey')
    axis.text(0, 16, 'Iteration 0', fontsize=8)
    axis.text(3, 8, f'Iteration {len(loss_his)}', fontsize=8)
    axis.set_xlabel("Weight")
    axis.set_ylabel("Cost")
    plt.show()

plot_cost_w_grad(w_his, loss_his)


###Comparacion de optimizadores

NAMES = ['SGD', 'SGD_momentum', 'Adam']

(xtrain, ytrain), (xtest, ytest) = fashion_mnist.load_data()

xtrain = np.reshape(xtrain, (len(xtrain), -1))
xtest = np.reshape(xtest, (len(xtest), -1))

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

#Normalizacion
xtrain = np.apply_along_axis(lambda x: x/255, 1, xtrain)
xtest = np.apply_along_axis(lambda x: x/255, 1, xtest)

# validation set
index = 50000
xval, yval = xtrain[index:], ytrain[index:]
xtrain, ytrain = xtrain[:index], ytrain[:index]
print(xtrain.shape, xval.shape, xtest.shape)

#Creamos un modelo con 2 capas con acticador "relu", un dropout, otra capa final con softmax
model = Sequential(
[
    Dense(128, 
          input_shape=xtrain.shape[1:],
          activation='relu', 
          name='dense_1'),
    Dense(64, 
          activation='relu', 
          name='dense_2'),
    Dropout(0.2),
    Dense(10, 
          activation='softmax', 
          name='dense_3')
], name='Sequential')

model.build()

#Muestra un resumen en una cuadricula del modelo
print(model.summary())

epochs = 10
batch_size = 64
shuffle = True

# dicts for storing results
loss      = {opt:[] for opt in NAMES}
val_loss  = {opt:[] for opt in NAMES}
acc       = {opt:[] for opt in NAMES}
val_acc   = {opt:[] for opt in NAMES}
test_acc  = {}
test_loss = {}

weights = model.get_weights().copy()

#Aplica diferentes optimizadores a los datos de entrenamiento
#Lo configura y luego lo fitea
with tqdm(desc='Training', total=len(NAMES*epochs)) as pbar:
    for name in NAMES:
        optimizer=''
        
        # prepare model
        model.set_weights(weights)
        if name == 'SGD':
            optimizer= SGD(learning_rate=0.001)
        elif name=='SGD_momentum':
            optimizer=SGD(learning_rate=0.001, momentum=0.9)
        elif name=='Adam':
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        print('Optimizer: ', name)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        
        # train model
        for epoch in range(epochs):
            his = model.fit(xtrain, ytrain,
                            epochs=1,
                            batch_size=batch_size,
                            validation_data=(xval, yval),
                            shuffle=shuffle,
                            verbose=0)
            
            # update dictionaries
            loss[name].append(his.history['loss'][0])
            val_loss[name].append(his.history['val_loss'][0])
            acc[name].append(his.history['acc'][0])
            val_acc[name].append(his.history['val_acc'][0])   
            pbar.update(1)
            
        # inference
        t_loss, t_acc = model.evaluate(xtest, ytest, verbose=0)
        test_loss[name] = t_loss
        test_acc[name] = t_acc

fig, axs = plt.subplots(2, 2, figsize=(13, 7))
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

#Ploteamos 4 graficos, cada uno con el error de los modelos y la exactitud
#El mejor es Adam en todos los graficos y el peor es Gradiente solo
for index, result, title in zip([[0, 0], [0, 1], [1, 0], [1, 1]], 
                                [loss, val_loss, acc, val_acc], 
                                ['loss', 'val_loss', 'acc', 'val_acc']):
    i, j = index
    for name, values in result.items():
        axs[i, j].plot(values, label=name)
        axs[i, j].set_title(title, size=15)
        axs[i, j].set_xticks([e for e in range(epochs)])
        axs[i, j].legend(loc="best", prop={'size': 10})
plt.show()

w, h = 28, 28

#Tomamos algunas muestras de etiquetas
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

#Hace predicciones el modelo
y_hat = model.predict(xtest)

#Da predicciones de que es la imagen, le achunta a la mayoria, pero se equivoco en uno el mas dificil
#diferenciar una sandalia de una zapatilla
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(range(16)):
    
    ax = figure.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(xtrain.reshape(xtrain.shape[0], w, h, 1)[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(ytest[index])
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index],
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
plt.show()
 

#######################################################################################################################

                                       #Optimizador para keras
                                 #Los ejemplos estan llenos de errores
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tensorflow INFO and WARNING messages are not printed
# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from tqdm import tqdm
import numpy as np

import tensorflow as tf
import keras
import sklearn

print(tf.__version__)
print(sklearn.__version__)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
#from keras.wrappers.scikit_learn import KerasClassifier

import skillsnetwork

# Vectorize integer sequence
def vectorize_sequence(sequence, dimensions):
    results = np.zeros((len(sequence), dimensions))
    for index,value in enumerate(sequence):
        if max(value) < dimensions:
            results[index, value] = 1
    return results

# Convert label into one-hot format
def one_hot_label(labels, dimensions):
    results = np.zeros((len(labels), dimensions))
    for index,value in enumerate(labels):
        if value < dimensions:
            results[index, value] = 1
    return results

X = np.load("x.npy", allow_pickle=True)
y = np.load ("y.npy", allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Es un diccionario de palabras indice
word_to_ind = tf.keras.datasets.reuters.get_word_index(path="reuters_word_index.json")

    
dim_x = max([max(sequence) for sequence in X_train])+1
dim_y = max(y_train)+1

X_train_vec = vectorize_sequence(X_train, dim_x)
X_test_vec = vectorize_sequence(X_test, dim_x)
y_train_hot = one_hot_label(y_train, dim_y)
y_test_hot = one_hot_label(y_test, dim_y)



#Funcion para crear un modelo
def create_model(neurons = 10, **kwargs):
    model = Sequential()
    model.add(Dense(10, activation='linear'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(46, activation='softmax'))
    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn=None, **kwargs):
        self.build_fn = build_fn
        self.model_params = kwargs
        self.model = None

    def set_params(self, **params):
        self.model_params.update(params)
        return self

    def get_params(self, deep=True):
        return self.model_params

    def fit(self, X, y, **fit_params):
        self.model = self.build_fn(**self.model_params)
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

    def predict_proba(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]


np.random.seed(0)

#Creamos un modelo, les especificamos el bath y epochs
model = KerasClassifierWrapper(build_fn=create_model)

print(X_train_vec.shape, y_train_hot.shape)

#Ajustamos el modelo, y sacamos la puntuacion
model.fit(X_train_vec, y_train_hot, epochs=3, batch_size=10)
initial_score = model.score(X_test_vec, y_test_hot)
print(f"Initial test accuracy: {initial_score:.4f}")


#base_score = model.evaluate(X_test_vec, y_test_hot)
#print('Test loss:', base_score[0])
#print('Test accuracy:', base_score[1])


batch_size = [10, 20, 60, 80]
epochs = [1, 3, 5]
neurons = [1, 10, 20, 30]

params = dict(batch_size=batch_size, epochs=epochs, neurons=neurons)
print(params)


search = RandomizedSearchCV(estimator=model, param_distributions=params, cv=3, scoring='accuracy', n_jobs=-1)



search_result = search.fit(X_train_vec, y_train_hot)

best_params = search_result.best_params_
keras_clf = KerasClassifierWrapper(build_fn=create_model, **best_params)

means = search_result.cv_results_['mean_test_score']
stds = search_result.cv_results_['std_test_score']
params = search_result.cv_results_['params']

print("Best mean cross-validated score: {} using {}".format(round(search_result.best_score_,3), search_result.best_params_))

for mean, stdev, param in zip(means, stds, params):
    print("Mean cross-validated score: {} ({}) using: {}".format(round(mean,3), round(stdev,3), param))

print("Best test score: %.3f" % search_result.best_estimator_.score(X_test_vec, y_test_hot))




def create_model(optimizer = 'RMSprop', learning_rate = 0.1, dropout_rate = 0.2):
          model = Sequential([                 Dense(64, activation='linear'),                 Dropout(0.2),                Dense(64,activation='relu'),                Dense(46,activation='softmax')])
          model.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=["accuracy"])
          return model
          
#def create_model(model__optimizer = 'RMSprop', model__learning_rate = 0.1, model__dropout_rate = 0.2, init='glorot_uniform'):
#          model = Sequential()
#          model.add(Dense(64, kernel_initializer=init, activation='linear'))
#          model.add(Dropout(0.2))
#          model.add(Dense(64, kernel_initializer=init,activation='relu'))
#          model.add(Dense(46, kernel_initializer=init,activation='softmax'))
#          model.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=["accuracy"])
#          return model         
          
#model.fit(X_train_vec, y_train_hot, batch_size=100, epochs=1)
np.random.seed(0)
model = KerasClassifier(model=create_model, verbose=0, batch_size=100, epochs=1)
#model.fit(X_train_vec, y_train_hot)
#base_score = base_model.score(X_test_vec, y_test_hot)
#print("The baseline accuracy is: {}".format(base_score))

#optimizer = ['SGD','RMSprop','Adam']
#learning_rate = [0.01, 0.1, 1]
#dropout_rate = [0.1, 0.3, 0.6, 0.9]
#params = dict(model__optimizer=optimizer, optimizer__learning_rate=learning_rate, dropout_rate = dropout_rate)

params = {
    'model__optimizer': ['SGD', 'RMSprop', 'Adam'],
    'model__optimizer__learning_rate': [0.01, 0.1, 1],
    'model__rate': [0.1, 0.3, 0.6, 0.9]
}

search = RandomizedSearchCV(estimator=model, param_distributions=params, cv=3)
search_result = search.fit(X_train_vec, y_train_hot)

print("Best mean cross-validated score: {} using {}".format(round(search_result.best_score_,3), search_result.best_params_))
print("Best test score: %.3f" % search_result.best_estimator_.score(X_test_vec, y_test_hot))


#######################################################################################################################

                                     #Funcion de perdida categorica, entropia

#Primero destacar la funcion softmax, que generalmente se ocupa como una funcion de salida sobre todo en redes neuronales
#convolucionales, se ocupa para normalizar la salida y las probabilidades

#En el caso de imagenes muestra probabilidades de ser cierto cosa



import cv2
from urllib.request import urlopen
from PIL import Image
import IPython
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd, numpy as np
from keras.datasets import mnist, fashion_mnist
import random
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.losses import CategoricalCrossentropy,SparseCategoricalCrossentropy,BinaryCrossentropy
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions
import tensorflow as tf
print(tf.__version__)
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import PIL

def generate_multiclass_blobs(num_samples_total, training_split, cluster_centers, num_classes, loss_function_used):
    X, targets = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 1.5)
    categorical_targets = to_categorical(targets)
    X_training = X[training_split:, :]
    X_testing = X[:training_split, :]
    Targets_training = categorical_targets[training_split:]
    Targets_testing = categorical_targets[:training_split].astype(np.int32)
    return X_training, Targets_training, X_testing, Targets_testing


def generate_binary_blobs(num_samples_total, training_split, loss_function_used):
    X, targets = make_blobs(n_samples = num_samples_total, centers = [(0,0), (15,15)], n_features = 2, center_box=(0, 1), cluster_std = 2.5)
    targets[np.where(targets == 0)] = -1
    X_training = X[training_split:, :]
    X_testing = X[:training_split, :]
    Targets_training = targets[training_split:]
    Targets_testing = targets[:training_split]
    return X_training, Targets_training, X_testing, Targets_testing


#Crea cluster solo con numero, tambien dice cuales son los centros
num_samples = 1000
test_split = 250
cluster_centers = [(15,0), (30,15)]
num_classes = len(cluster_centers)
loss_function_used = BinaryCrossentropy(from_logits=True)

#Segun la catnidad de muestras, las separaciones, la funcion de perdida crea 2 cluster
X_training, Targets_training, X_testing, Targets_testing=generate_binary_blobs(num_samples, test_split, loss_function_used)

plt.figure(figsize=(4, 4))
plt.scatter(X_training[:,0], X_training[:,1])
plt.title('Linearly separable data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

feature_vector_shape = X_training.shape[1]
input_shape = (feature_vector_shape,)

#Hacemos un modelo esa es la gracia de tomar las dimencioens de los datos de entrenamiento
model = Sequential()
model.add(Dense(12, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss=loss_function_used, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
history = model.fit(X_training, Targets_training, epochs=30, batch_size=5, verbose=1, validation_split=0.2)

#Muestra el resultado, logra diferenciar los 2 cluser
test_results = model.evaluate(X_testing, Targets_testing, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

#Plotea, las deciciones en las cual tomo el modelo la prediccion
plot_decision_regions(X_testing, Targets_testing, clf=model, legend=2)
plt.figure(figsize=(4, 4))
plt.show()


#Ahora con 3 clases
num_samples = 1000
train_split = 250
cluster_centers = [(-10, 5), (0, 0), (10, 5)]
num_classes = len(cluster_centers)
loss_function_used = CategoricalCrossentropy(from_logits=True)

#Crea 3 cluster
X_training, Targets_training, X_testing, Targets_testing= generate_multiclass_blobs(num_samples, train_split,
              cluster_centers, num_classes,
              loss_function_used)

plt.scatter(X_training[:,0], X_training[:,1])
plt.title('Linearly separable data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

feature_vector_shape = X_training.shape[1]
input_shape = (feature_vector_shape,)

#Creamos un nuevo modelo
model = Sequential()
model.add(Dense(12, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=loss_function_used, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_training, Targets_training, epochs=30, batch_size=5, verbose=1, validation_split=0.2)

#Vemos los resultados
test_results = model.evaluate(X_testing, Targets_testing, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

class Onehot2Int(object):

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        y_pred = self.model.predict(X)
        return np.argmax(y_pred, axis=1)

# fit keras_model
keras_model_no_ohe = Onehot2Int(model)

#Logra un buena prediccion de los 3 cluster
plot_decision_regions(X_testing, np.argmax(Targets_testing, axis=1), clf=keras_model_no_ohe, legend=3)
plt.show()


###Ejemplo 2

#Tomamos enseguida imagenes de numeros
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

#Mostramos tamaños
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

#Un grafico para mostrar 4 imagenes
plt.rcParams['figure.figsize'] = (6,6) 

for i in range(4):
    plt.subplot(2,2,i+1)
    num = random.randint(0, len(X_train))
    plt.imshow(X_train[num], cmap='gray', interpolation='none')
    plt.title("class {}".format(y_train[num]))
    
plt.tight_layout()
plt.show()

#Cambiamos de dimencion las imagenes
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]* X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

print(X_train.shape)
print(X_test.shape)

#Para escalar se pasan a tipo float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(X_train)

X_train /= 255
X_test /= 255

print(X_train)
print(y_train)

enc = OneHotEncoder(sparse_output=False)
enc.fit(y_train.reshape(-1, 1))
print(enc.categories_)

#Estan escalados y separados por clasificacion
y_train_enc = enc.transform(y_train.reshape(-1,1))
y_test_enc = enc.transform(y_test.reshape(-1,1))

#Para crear el modelo
feature_vector_shape = X_train.shape[1]
input_shape = (feature_vector_shape,)
num_classes = 10
loss_function_used=CategoricalCrossentropy(from_logits=True)
model = Sequential()
model.add(Dense(12, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=loss_function_used, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
#Se demora bastante en las iteraciones porque son demaciadas imagenes
history = model.fit(X_train, y_train_enc, epochs=25, batch_size=5, verbose=1, validation_split=0.2)

#tuvo un 94% de reconocimiento de numeros en las imagenes
test_results = model.evaluate(X_test, y_test_enc, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

###Ejemplo 3

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/images/house_number_5.jpeg'
image = Image.open(urlopen(URL)).convert('RGB')
print(image)

#Creamos un modelo
feature_vector_shape,input_shape = 784,784
num_classes = 10
loss_function_used = CategoricalCrossentropy(from_logits=True)
model = Sequential()
model.add(Dense(12, input_shape=(input_shape,), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=loss_function_used, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_train, y_train_enc, epochs=25, batch_size=5, verbose=1, validation_split=0.2)

#Cargamos una imagen aparte
img_rows, img_cols = 28, 28
img = Image.fromarray(np.uint8(image)).convert('L')
img_gray = img.resize((img_rows, img_cols), PIL.Image.Resampling.LANCZOS)
print(img_gray)

arr = np.array(img_gray)
arr = arr.reshape((img_cols*img_rows))
arr = np.expand_dims(arr, axis=0)

#Muestra una prediccion
prediction = model.predict(arr)
print(np.argmax(prediction))


###Ejemplo 5

#Cargamos la data de mnist fashion de ropa con datos de entrenamiento
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#Mostramos 4 imagenes de las data fashion
plt.rcParams['figure.figsize'] = (6,6) 

for i in range(4):
    plt.subplot(2,2,i+1)
    num = random.randint(0, len(X_train))
    plt.imshow(X_train[num], cmap='gray', interpolation='none')
    plt.title("class {}".format(y_train[num]))
    
plt.tight_layout()
plt.show()

#Cambimos las dimenciones
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]* X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

print(X_train.shape)
print(X_test.shape)

#Tipo float para escalar
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

#Codificamos tambien
enc = OneHotEncoder(sparse_output=False)
enc.fit(y_train.reshape(-1, 1))
print(enc.categories_)
y_train_enc = enc.transform(y_train.reshape(-1,1))
y_test_enc = enc.transform(y_test.reshape(-1,1))

feature_vector_shape = X_train.shape[1]
input_shape = (feature_vector_shape,)
num_classes = 10
loss_function_used = CategoricalCrossentropy(from_logits=True)

#Creamos un nuevo modelo
model = Sequential()
model.add(Dense(12, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=loss_function_used, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_train, y_train_enc, epochs=25, batch_size=5, verbose=1, validation_split=0.2)

#Da un resultado de exactitud de 0.84 con la funcion de perdida CategoricalCrossentropy
test_results = model.evaluate(X_test, y_test_enc, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

#Probamos otro modelo con la funcion de perdida sparse
feature_vector_shape = X_train.shape[1]
input_shape = (feature_vector_shape,)
num_classes = 10
loss_function_used = SparseCategoricalCrossentropy()

model = Sequential()
model.add(Dense(12, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=loss_function_used, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=25, batch_size=5, verbose=1, validation_split=0.2)

#Alcanzo una exactitud de 0.86 bastante parecida con la anterior
test_results = model.evaluate(X_test, y_test_enc, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')


#######################################################################################################################

                                           #Imagen Convolucional
                                             #Operadores y DoG

#Un red neuronal convolucional, es una red dedicada a procesar imagenes, funciona moviendo un kernel de cierto tamaño
#pequeño que recorre la imagen, viendo caracterisitcas de sus pixeles
#Este kernel hace calculos sobre la imagen, esos calculos logra crear una nueva imagen con los resultados, a eso
#se le llama convolucion, esto se puede hacer reiteradas veces
#Al hacer esto puede encontrar caracteristicas vertical o horizontales en la imagen

#Los kernel fueron creados para reconocer bordes en las imagenes

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tensorflow INFO and WARNING messages are not printed 

import pathlib
import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module3/L1/flower_photos.tgz"
data_dir = keras.utils.get_file(origin=dataset_url,
                                fname='flower_photos',
                                untar=True)

data_dir = pathlib.Path(data_dir)

for folder in data_dir.glob('[!LICENSE]*'):
    print('The', folder.name, 'folder has',
          len(list(folder.glob('*.jpg'))), 'pictures')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count, 'total images')

pics = list()
pics_arr = list()
p_class = list()

img_width = 300
img_height = 300

#Plot de las imagenes, contiene una data de solo imagenes de flores
plt.figure(figsize=(20,5))
for idx, folder in enumerate(data_dir.glob('[!LICENSE]*')):
    cat = list(data_dir.glob(folder.name + '/*'))
    pic = PIL.Image.open(str(cat[0])).resize((img_width, img_height))
    pic_arr = np.array(pic)
    clss = folder.name
    
    plt.subplot(1,5,idx+1)
    plt.imshow(pic)
    plt.title(clss)
    plt.axis('off')
    
    pics.append(pic)
    pics_arr.append(pic_arr)
    p_class.append(clss)
plt.show()

img = pics[2]
print(img)


def v_grad(shape, dtype=None):
    # Here we use a single numpy array to define our x gradient kernel
    grad = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]).reshape((3, 3, 1, 1))
    # this line is quite important, we are saying we want one 3x3 kernel each for one channel of pixels (grayscale)
    
    # We check to make sure the shape of our kernel is the correct shape
    # according to the initialization of the Convolutional layer below
    assert grad.shape == shape
    return keras.backend.variable(grad, dtype='float32')

def h_grad(shape, dtype=None):
    grad = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
        ]).reshape((3, 3, 1, 1))
    
    assert grad.shape == shape
    return keras.backend.variable(grad, dtype='float32')

#Implementando la deteccion de bordes

#Creamos nuestra red convolucional
input_layer = layers.Input(shape=(img_width, img_height, 1))

h_conv = layers.Conv2D(filters=1, # the number of kernels we are using, kernel and filter are interchangeable terms
                       kernel_size=3,
                       kernel_initializer=h_grad,
                       strides=1,
                       padding='valid')  # 'valid' means no padding

v_conv = layers.Conv2D(filters=1,
                       kernel_size=3,
                       kernel_initializer=v_grad,
                       strides=1,
                       padding='valid')

#Dos modelos con diferntes kernel
h_model = keras.Sequential([input_layer, h_conv])
v_model = keras.Sequential([input_layer, v_conv])

print(h_model.summary())
print(v_model.summary())

#tomamos una imagen en una escala de grises
gray = ImageOps.grayscale(img)

#Le cambiamos las dimencioens
input_img = np.array(gray).reshape((1, img_width, img_height, 1))

out_d= h_model.layers[0].output.shape[1:]

#Hacemos una prediccion solo de esa imagen con los dos modelos
Gx = h_model.predict(input_img).reshape(out_d)
Gy = v_model.predict(input_img).reshape(out_d)

#Operador prewitt o sobel
G = np.sqrt(np.add(np.multiply(Gx, Gx), np.multiply(Gy, Gy)))

plt.figure(figsize=(18, 12))
plt.subplot(2, 3, 1)
plt.imshow(img)
#Mostrmos la imagen original sin modificaciones
plt.title("Original Image")
plt.subplot(2, 3, 2)
plt.imshow(gray, cmap=plt.get_cmap('gray'))
#Le habiamos dicho en escala de grises
plt.title("Grayscale Image")
plt.subplot(2, 3, 4)
plt.imshow(Gx.astype('uint8'), cmap=plt.get_cmap('gray'))
#Con el kernel horizonal, la fila del medio con 0, es una imagen rugosa, que extrae rugorizades de la parte horizontal
plt.title("Horizontal Gradient")
plt.subplot(2, 3, 5)
plt.imshow(Gy.astype('uint8'), cmap=plt.get_cmap('gray'))

#Con el kernel vertical, la columna del medio con 0, es una imagen que rugosa, que logra extrarse rugosidades de la parte
#vertical, en la medida que puede
plt.title("Vertical Gradient")
plt.subplot(2, 3, 6)
plt.imshow(G, cmap=plt.get_cmap('gray'))

#Con un operador especial que combina los dos kernel, es como una imagen como borrosa, mucha mezcla
plt.title("Image after Sobel Operator")
plt.show()


###Ejemplo 2 Operados DoG

#Operadores especiales para los bordes, reconocen muy bien los bordes de las imagenes

#Blurred_sm y Blurred_lg son operadores y juntos, logran hacer DoG

sigma_sm = 5
sigma_lg = 9
blurred_sm = cv2.GaussianBlur(np.array(gray), (sigma_sm, sigma_sm), sigma_sm)
blurred_lg = cv2.GaussianBlur(np.array(gray), (sigma_lg, sigma_lg), sigma_lg)

DoG = blurred_sm - blurred_lg

plt.figure(figsize=(18, 12))
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.subplot(2, 3, 2)
plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.title("Grayscale Image")
plt.subplot(2, 3, 4)
plt.imshow(blurred_sm.astype('uint8'), cmap=plt.get_cmap('gray'))
plt.title("Blurred: Small Sigma")
plt.subplot(2, 3, 5)
plt.imshow(blurred_lg.astype('uint8'), cmap=plt.get_cmap('gray'))
plt.title("Blurred: Large Sigma")
plt.subplot(2, 3, 6)
plt.imshow(DoG.astype('uint8'), cmap=plt.get_cmap('gray'))
plt.title("Image after DoG Operator")
plt.show()


#######################################################################################################################

                                  #Stride, Activacion, Pooling y Padding
                                         #Redes convolucionales


import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
from itertools import accumulate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits, load_wine

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageOps
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, datasets
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D

sns.set_context('notebook')
sns.set_style('white')

def sobel(img, strides, padding, activation=None):
    
    input_layer = layers.Input(shape=(img_height, img_width, 1))

    v_conv = layers.Conv2D(filters=1,
                       kernel_size=3,
                       kernel_initializer=v_grad,
                       strides=strides,
                       padding=padding,
                       activation=None)
    h_conv = layers.Conv2D(filters=1, 
                   kernel_size=3,
                   kernel_initializer=h_grad,
                   strides=strides,
                   padding=padding,
                   activation=None)
    
    v_model = keras.Sequential([input_layer, v_conv])
    h_model = keras.Sequential([input_layer, h_conv])    
    
    out_d = h_model.layers[0].output.shape[1:]
    Gx = h_model.predict(img).reshape(out_d)
    Gy = v_model.predict(img).reshape(out_d)
    G = np.sqrt(np.add(np.multiply(Gx, Gx), np.multiply(Gy, Gy)))
    
    return G


###Padding y Stride

#Las redes convolucionales pierden mucha informacion al aplicar kernel, esto puede ser perjudicial, puesto que de perder
#informacion de los bordes, no lograra reconocer nada mas.
#Padding y Stride lograr recuperar esa informacion a medida que se le agregan mas convoluciones

###Padding (relleno)

#Padding logra que la imagen tenga las mismas dimenciones apesar de usar convoluciones



from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters = 1,
                 kernel_size = (3,3),
                 padding = 'same',
                 input_shape = (10, 10, 1)))
print(model.summary())

###Stride (zancada)

#Stride hace un calculo especial, para reducir la imagen a lo que nosotros le especifiquemos

model = Sequential()
model.add(Conv2D(filters = 1,
                 kernel_size = (3, 3),
                 strides = (2, 2),
                 padding = "same",
                 input_shape = (3, 3, 1)))
print(model.summary())



#Creando Kernels
input_ = np.array([[1, 1, 3],
              [2, 1, 2],
              [3, 1, 4]]).reshape(1, 3, 3, 1)

kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]]).reshape(3, 3, 1, 1) # reshaping the kernel is important

b = np.array([0.0])

model.set_weights([kernel, b])
output_ = model.predict(input_)

for r in range(output_.shape[1]):
    print([output_[0,r,c,0] for c in range(output_.shape[2])])

def v_grad(shape, dtype=None):
    # Here we use a single numpy array to define our x gradient kernel
    grad = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]).reshape((3, 3, 1, 1))
    # this line is quite important, we are saying we want one 3x3 kernel each for one channel of pixels (grayscale)
    
    # We check to make sure the shape of our kernel is the correct shape
    # according to the initialization of the Convolutional layer below
    assert grad.shape == shape
    return keras.backend.variable(grad, dtype='float32')

def h_grad(shape, dtype=None):
    grad = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
        ]).reshape((3, 3, 1, 1))
    
    assert grad.shape == shape
    return keras.backend.variable(grad, dtype='float32')

#Descarmos la imagen

img_width = 350
img_height = 500
img = PIL.Image.open("pisa.jpg").resize((img_width, img_height))
print(img)

input_img = ImageOps.grayscale(img)
input_img = np.array(input_img).reshape((1, img_height, img_width, 1))

fig, axs = plt.subplots(1, 3, figsize=(9, 10), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0.1)

#El efecto de padding con diferentes stride, logra separar los objetos del fondo, identificandolos
for i, ax in enumerate(axs.flat):
    output = sobel(img = input_img, 
                   padding='same',
                   strides=i+1).astype('int').clip(0,255)

    ax.imshow(output, cmap='gray')
    ax.set_title(f"Strides: {i+1}, Shape: {output.shape}", fontsize=13)
    ax.axis('off')
plt.show()

###Activacion

#Las activaciones se ocupan para ponerle restricciones a los valores de los pixeles, en los calculos o incluso
#de base pueden tener valores, menores a 0 y mayores a 255, el activador le pone restricciones para que no superen
#o bajen estos valores, ademas agregan no linealidad al modelo para que aprenda patrones complejos

#La funcion sigmoid se llama tambien funcion de aplastamiento porque solo deja que los valores esten entre 0 y 1
#Relu convierte todos los valores menores a 0 en 0


def sigmoid(X):

    return 1/(1 + np.exp(-X))

X = np.linspace(-10, 10, 100)
sigmoid_X = sigmoid(X)
plt.plot(X, sigmoid_X)
plt.axhline(y=0.5, color='r', linestyle='-')
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")

def relu(X):
    return np.maximum(0, X)

X = np.linspace(-10, 10, 100)
relu_X = relu(X)
plt.plot(X, relu_X)
plt.xlabel("x")
plt.ylabel("Relu(x)")

#Se descarga la imagen

img_width = 500
img_height = 280

img = Image.open("lambor.jpeg").resize((img_width, img_height))
print(img)

arr = np.array(img)
red_c = arr[:,:,0]
green_c = arr[:,:,1]
blue_c = arr[:,:,2]

channels = [red_c, green_c, blue_c]
names = ["Red", "Green", "Blue"]

plt.figure(figsize=(18, 5))

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(channels[i], cmap='gray')
    plt.axis("off")
    plt.title(f"{names[i]} channel", fontsize=13)
plt.show()

kernel = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]]).reshape(3,3,1,1)
b = np.array([0.0])

#Hacemos el modelo convolucional, con padding y activacion relu
#relu en si no hace nada solo los pone en negro muchas caracterisitcas al ser menores que 0
model = keras.Sequential()
model.add(layers.Conv2D(input_shape = (img_height, img_width, 1),
                 filters=1, 
                 kernel_size=3, 
                 padding='same',
                 activation='relu'
                 ))
model.set_weights([kernel, b])
print(model.summary())


acts = []
plt.figure(figsize=(18,5))

for i in range(3):
    plt.subplot(1,3,i+1)
    
    # loop through each channel
    input_ = channels[i].reshape((1, img_height, img_width, 1)) 
    act = model.predict(input_).squeeze(0).squeeze(2).astype('int').clip(0,255)
    # store the result in a list called "acts"
    acts.append(act)
    
    plt.imshow(act, cmap='gray')
    plt.axis("off")
    plt.title(f"{names[i]} Channel with activation", fontsize=13)
arr_hat = np.dstack((acts[0],acts[1],acts[2]))

plt.figure(figsize=(10,5))
plt.imshow(arr_hat)
plt.axis("off")
plt.show()

###Pooling (agrupacion)

#El pooling es un metodo para reducir la huella de memoria de todos los calculos

#Hay dos tipos de pooling, pooling maximo o pooling promedio

#Comunmente se ponen entre cada convolucion y luego se aplicar una activacion

###Max Pooling

#Lo que hace es aplicar un especie de kernel que recorre la imagen, en cada avance de kernel
#se queda con el valor mas alto, esto reduce la imagen a menos datos, dependiendo de la ventana
#de kernel que ocupemos
#Esto es lo que hace es quedarse con las caracterisitcas mas importantes, es util cuando existe un fondo
#muy oscuro, porque podemos quitar ese fondo y quedarnos solo con elemento claro, con en la data de
#reconocer letras o numeros en la imagen
#Lo malo de max pooling es que tienda a blanquear la imagen

###Promedio Pooling

#De el kernel el promedio de pooling lo que hace es calcular un promedio del kernel que estemos ocupando
#esto es util para suavizar la imagen cuando tiene mucho ruido, pero pierde calidad, no hace un
#efecto fuerte a las imagenes

#Descargamos la data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print("MNIST downloaded!")
print("Train set shape:", X_train.shape)

images = []
plt.figure(figsize=(15,3))

#Mostramos unas imagenes
for i in range(5):
    img = X_train[np.random.randint(0, 60000)].astype('float')
    images.append(img)
    
    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.axis('off')  
plt.show()

#Creamos dos modelos con los pooling
max_pool = keras.Sequential([MaxPooling2D(pool_size = (2,2))])
avg_pool = keras.Sequential([AveragePooling2D(pool_size = (2,2))])

#Los ponemos como objetos ploteables
fig1, axs1 = plt.subplots(1, 5, figsize=(12,3), constrained_layout=True)
fig2, axs2 = plt.subplots(1, 5, figsize=(12,3), constrained_layout=True)
fig1.suptitle("Max pooling result", fontsize=20)
fig2.suptitle("Average pooling result", fontsize=20)

#Ploteamos los resultados
for img, ax1, ax2 in zip(images, axs1.flat, axs2.flat):
    input_ = img.reshape(1, 28, 28, 1)
    ax1.imshow(max_pool.predict(input_).squeeze(0).squeeze(2))
    ax1.axis('off')
    ax2.imshow(avg_pool.predict(input_).squeeze(0).squeeze(2))
    ax2.axis('off')
plt.show()


###Ejemplos 

from keras.utils import to_categorical
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential

#Funcion que define una data al azar desde cifa10
def load_cifar10():
    (trainX, trainY), (testX, testY) = datasets.cifar10.load_data()
    
    trainX = trainX.astype('float32') / 255
    testX = testX.astype('float32') / 255
    
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    
    return trainX, trainY, testX, testY

#Tambien cargamos una muestra para saber cuantos vienen
X_train, y_train, X_test, y_test = load_cifar10()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10,10))

#Muestra 25 imagenes, las primeras 25
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(class_names[np.where(y_train[i]==1)[0][0]])
    plt.axis("off")
plt.show()    

Conv = layers.Conv2D(filters=32,
                     kernel_size=(3,3),
                     kernel_initializer='he_uniform',
                     padding='same',
                     activation='relu')

Max = layers.MaxPooling2D(pool_size=(2,2))

#Creamos un modelo con convoluciones y maxpooling
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', 
                 input_shape=(32, 32, 3))) 
# Don't forget specifying input_shape in the 1st Conv2D layer of your CNN

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
print(model.summary())
# compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model = model.fit(X_train, y_train, epochs=25, batch_size=5, verbose=1, validation_split=0.2)

pred = model.predict(X_test)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_test[i])
    plt.title(pred[i])
    plt.axis("off")
plt.show()    


#######################################################################################################################
                                                        
                                            #Componentes de red neurnal convolucional
                                            #Construccion de filtros, capas, aplanado
                                                       #y campo receptivo
 

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tensorflow INFO and WARNING messages are not printed 

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import PIL
from PIL import Image, ImageOps
import tensorflow as tf

import glob
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Input

def calc_rf(model):
    # Initialize an array storing all the layers' receptive field sizes and set layer 0 (input) RF to 1
    num_layers = len(model.layers)
    rf_arr = np.empty(num_layers+1, dtype=int)
    rf_arr[0] = 1
    # Initialize an array storing all the layers' jump sizes and set layer 0 (input) jump to 1
    j_arr = np.empty(num_layers+1, dtype=int)
    j_arr[0] = 1
    
    for i in range(num_layers):
        layer = model.layers[i]
        k = layer.kernel_size[0]
        s = layer.strides[0]
        j_in = j_arr[i]
        j_out = j_in * s
        r_out = rf_arr[i] + (k - 1) * j_in
        j_arr[i+1] = j_out
        rf_arr[i+1] = r_out        
        print("Layer {}: {} \n Jump Size: {}\n Effective Receptive Field Size: {}".format(i+1, layer.name, j_arr[i+1], rf_arr[i+1]))
        print("------")


#Porque agregar capas a una red convolucional?

#Una CNN se comporta como un ojo humano, cada neurona se encanrga de ver una parte especifica d ela
#vista, en CNN cada output de los ojos recopila informacion de entrada a nuestras neuronas, recopilando asi
#cada caracteristica, nuestras neuronas tienen un unica salida que nos dira que objeto es. Esta region que recopila
#los datos se le llama campo receptivo de la caracterisitca

#A medida que se le agregan mas capas, se le llama tambien campo receptivo efectivo, dado que cada convolucion
#genera nuevas caracterisitcas 
#Hay una formula para calcula el campo receptivo efectivo

###CAMPO RECEPTIVO EFECTIVO (ERF)

#El campo efectivo receptivo describe cuanta informacion aprende la red, cuanto mas aprenda podra ver mejores
#cosas
#Si lo vemos como la analogia del ojo, entre mas neuronas mejor, mas informacion aprenderemos
#En CNN no se puede aprender infinito, solo es necesario aprender lo que cubra toda la imagen, tener mas neuronas
#que lo que neesita imagen, hara que no tengan ninguna utilidad

#Cosas que pueden afectar ERF

#Los pixeles de la imagen, que tenga demaciada poca resolucion, reduce las posibilidades de ocupar kernels
#La cantidad de convoluciones depende del kernel que ocupemos para hacer las convoluciones
#El numero de la Stride si es grande no podremos tener tanta informacion
#El pooling provoca perdida de resolucion, debido a que solo tomara la informacion importante, perderemos
#resolucion
#La tasa de dilatacion agrega valores entre los kernels de modo que no se apliquen muestras adyacentes

#Descargamos una imagen

img_width = 300
img_height = 300

dot = PIL.Image.open("RF_dot.png")
dot = ImageOps.grayscale(dot)
# dot = PIL.ImageOps.invert(dot)
dot = dot.resize((img_width, img_height))
plt.imshow(dot, cmap="gray")
plt.show()

#Definimos el kernel
kernel = (1/9)*np.ones((3,3,1,1))
b = np.array([0.0])

#Efecto sin nada en particular
model1 = Sequential()
model1.add(Conv2D(input_shape = (img_width, img_height, 1),
                 filters=1, 
                 kernel_size=(3,3)
                 ))

model1.layers[0].set_weights([kernel,b])

#Calcula el ERF=3
print(calc_rf(model1))

#Efecto con un stride de 2
model2 = Sequential()
model2.add(Conv2D(input_shape = (img_width, img_height, 1),
                 filters=1, 
                 kernel_size=(3,3), 
                 padding='same',
                 strides=2
                 ))

model2.layers[0].set_weights([kernel,b])

#Calcula el ERF=3, no cambio debido de que depende en gran medida del tamaño del kernel y de RF orginal
print(calc_rf(model2))

#Agregando muchas mas capas de convoluciones
model3 = Sequential()

model3.add(Conv2D(input_shape = (img_width, img_height, 1),
                 filters=1, 
                 kernel_size=(3,3)
                 ))
model3.add(Conv2D(filters=1, 
                 kernel_size=(3,3), 
                 strides=4
                 ))
model3.add(Conv2D(filters=1, 
                 kernel_size=(3,3), 
                 strides=5
                 ))

model3.layers[0].set_weights([kernel,b])
model3.layers[1].set_weights([kernel,b])
model3.layers[2].set_weights([kernel,b])

#Aumento mucho a ERF=13
print(calc_rf(model3))

models = [model1, model2, model3]
dot_tensor = np.array(dot).reshape((1,img_width,img_height,1))
fig, ax = plt.subplots(1,len(models))

#Revisando output
#Modelo 1 sigue igual, modelo2 sigue igual, modelo3 se volvio muy borrosa la imagen
for i in range(len(models)):
    plt.subplot(1,len(models),i+1)
    output = models[i].predict(dot_tensor)
    output = output.reshape(output.shape[1],output.shape[2])
    plt.title("Model {}".format(i+1))
    plt.imshow(output, cmap='gray')
plt.show()

###Ejemplo 2

img_width = 300
img_height = 300

image = PIL.Image.open("channel_image.jpg").resize((img_width, img_height))
print(image)

def edge_grad(shape, dtype=None):
    grad = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
        ]).reshape(shape)
    
    return grad

def sharpen_grad(shape, dtype=None):
    grad = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]).reshape(shape)
    
    return grad


###Una entrada multiples salidas

#Se refiere a la cantidad de filtros que se le pueden aplicar a una imagen
#En este ejemplo se le aplican dos kernel por lo tanto, hay dos salidas, uno con un filtro muy oscuro
#y otro mas gris

gray = ImageOps.grayscale(image)
img_tensor = np.array(gray).reshape((1, img_width, img_height, 1)) # Convert to a tensor for model prediction
print(gray)

#Definir kerneles, algo importante es que en este caso habran 2 filtros por lo tanto son 2 difrentes kerneles
#el ultimo numero del kernel indica que filtro es, los dos se le aplican a una entrada
kernels = np.array([sharpen_grad((3,3,1,1)), edge_grad((3,3,1,1))])
#este sera el segundo kernel que se le aplique a la entrada
kernels = kernels.reshape((3,3,1,2))
b = np.array([np.array([0.0]),np.array([0.0])]).reshape(2,)


model = Sequential([Input(shape=(300,300,1)),
Conv2D(filters = 2, 
                 kernel_size = (3, 3), 
                 padding='same', 
                 input_shape = (img_width, img_height, 1))])

model.set_weights([kernels,b])

print(model.summary())

layer_outputs = [layer.output for layer in model.layers] # Extracts the outputs of the layer

activation_model = Model(inputs=model.inputs, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

print(activation_model.summary())

activations = activation_model.predict(img_tensor) # Obtain the outputs (activations) from our model

filter_outputs = activations[0]  # Extract the outputs from the first (in this case, the only) layer of the model
print(filter_outputs.shape)

# Display weights of the first filter of this layer
print(filter_outputs[:,:,0])

# Display the filter outputs
names = ["Sharpening Filter", "Edge Filter"]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(filter_outputs[:,:,i], cmap='gray')
    plt.axis("off")
    plt.title(f"{names[i]}", fontsize=13)
plt.show()



###Multiple input, solo una salida

#Es cuando a una imagen RGB, sale por ejemplo solo en esala de grisses

img_tensor = np.array(image).reshape((1, img_width, img_height, 3))

kernel = np.array([sharpen_grad((3,3,1,1)), sharpen_grad((3,3,1,1)), sharpen_grad((3,3,1,1))])
kernel = kernel.reshape((3,3,3,1))
b = np.array([0.0])

model = Sequential()

model.add(Conv2D(filters = 1, 
                 kernel_size = (3, 3), 
                 padding='same', 
                 input_shape = (img_width, img_height, 3)))

model.set_weights([kernel,b])

model.summary()

output = model.predict(img_tensor)
output = output.reshape((img_width, img_height))

print(output)

plt.imshow(output, cmap="gray")
plt.show()


###Multiples entradas, multiples salidas

#En este caso ocupa 3 kernel y 2 filtros

img_tensor = np.array(image).reshape((1, img_width, img_height, 3))

kernel = np.zeros((3,3,3,2))

kernel[0] = np.dstack([sharpen_grad((3,3,1)),edge_grad((3,3,1))])
kernel[1] = np.dstack([sharpen_grad((3,3,1)),edge_grad((3,3,1))])
kernel[2] = np.dstack([sharpen_grad((3,3,1)),edge_grad((3,3,1))])

kernel = kernel.reshape((3,3,3,2))

b = np.array([np.array([0.0]),np.array([0.0])]).reshape(2,)

model = Sequential()

model.add(Conv2D(filters = 2, 
                 kernel_size = (3, 3), 
                 padding='same', 
                 input_shape = (img_width, img_height, 3)))

model.set_weights([kernel,b])

model.summary()

layer_outputs = [layer.output for layer in model.layers] 
activation_model = Model(inputs=model.inputs, outputs=layer_outputs) 
activations = activation_model.predict(img_tensor)
filter_outputs = activations[0]

#El resultado es una imagen por filtro en este caso solo muestra la primera
plt.imshow(filter_outputs[:,:,0])
plt.show()

#Los dos resultados son similares ya que los filtros estan hechos para los bordes
names = ["Sharpening Filter", "Edge Filter"]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(filter_outputs[:,:,i], cmap='gray')
    plt.axis("off")
    plt.title(f"{names[i]}", fontsize=13)
plt.show()


###Ejemplo de que sirve multiples capas?

#La gracia de tener multiples capas es que en cada capa podemos usarla para encontrar diferentes caracterisiticas 

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module3/L1/flower_photos.tgz"
data_dir = keras.utils.get_file(origin=dataset_url,
                                fname='flower_photos',
                                untar=True)

data_dir = pathlib.Path(data_dir)

for folder in data_dir.glob('[!LICENSE]*'):
    print('The', folder.name, 'folder has',
          len(list(folder.glob('*.jpg'))), 'pictures')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count, 'total images')

pics = list()
pics_arr = list()
p_class = list()

img_width = 300
img_height = 300

plt.figure(figsize=(20,5))
for idx, folder in enumerate(data_dir.glob('[!LICENSE]*')):
    cat = list(data_dir.glob(folder.name + '/*'))
    pic = PIL.Image.open(str(cat[0])).resize((img_width, img_height))
    pic_arr = np.array(pic)
    clss = folder.name
    
    plt.subplot(1,5,idx+1)
    plt.imshow(pic)
    plt.title(clss)
    plt.axis('off')
    
    pics.append(pic)
    pics_arr.append(pic_arr)
    p_class.append(clss)
plt.show()

#Creamos un modelo con multiples capas y kernels
classifier = Sequential()
classifier.add(Conv2D(3, (3, 3), padding='same', input_shape = (img_width, img_height, 3), activation = 'relu'))

classifier.add(Conv2D(2, (3, 3), activation='sigmoid'))
classifier.add(Conv2D(6, (5, 5), strides = 4, padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2))) 


###Aplanamiento

#El aplanamientos transformar una matriz pasarla a 1D, esto debido a que la salida solo permite una dimension
#Considerando que solo queremos una columna labels

#Añadimos al modelo anterior
classifier.add(Flatten()) 
classifier.summary()


###Capa Densa

#La ultima capa es aplanda, calcula el promedio ponderado y luego aplica la funcion de activacion, cada salida
#es una combinacion de todas las caracteristicas de entrada lo que hace que este completamente conectada

#Si tenemos 2 o mas clases debe ser una funcion de activacion softmax porque entrega probabilidades, entrega la
#clase que tenga mayor probabilidades para cada salida, recordando que son imagenes y necesita solo un labels

classifier.add(Dense(units = 512, activation = 'relu'))

classifier.add(Dense(units = 5, activation = 'softmax'))

classifier.summary()


######################################################################################################################

                                      #Construccion de red convolucional


import warnings
warnings.simplefilter('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pathlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image, ImageOps
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_activations_multilayer(num_layers, images_per_row, classifier, activations):
    layer_names = []
    for layer in classifier.layers[:num_layers]:
        layer_names.append(layer.name + ' layer')  # Names of the layers, so you can have them as part of your plot
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :,
                                                 col * images_per_row + row]
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        scale = 2. / size
        plt.figure(figsize=(scale*display_grid.shape[1],
                            scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

data_dir = Path("flower_photos")

for folder in data_dir.glob('[!LICENSE]*'):
    print('The', folder.name, 'folder has',
          len(list(folder.glob('*.jpg'))), 'pictures')

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count, 'total images')

img_width = 150
img_height = 150

batch_size = 64
epochs = 10

pics = list()
pics_arr = list()
p_class = list()

#Esto para separar las imagenes
plt.figure(figsize=(20,5))
for idx, folder in enumerate(data_dir.glob('[!LICENSE]*')):
    cat = list(data_dir.glob(folder.name + '/*'))
    pic = PIL.Image.open(str(cat[0])).resize((img_width, img_height))
    pic_arr = np.array(pic)
    clss = folder.name
    
    plt.subplot(1,5,idx+1)
    plt.imshow(pic)
    plt.title(clss)
    plt.axis('off')
    
    pics.append(pic)
    pics_arr.append(pic_arr)
    p_class.append(clss)
plt.show()

#Creamos una data de entrenamiento generada
train_gen = ImageDataGenerator(validation_split=0.2, 
                               rescale=1.0/255.0,
                                width_shift_range=0.2, # 0.1
                                height_shift_range=0.2, # 0.1
                                horizontal_flip=True)
train_set = train_gen.flow_from_directory(
                               directory=data_dir,
                               seed=10,
                               class_mode='sparse',
                               batch_size=batch_size,
                               shuffle=True,
                               target_size=(img_height, img_width),
                               subset='training')


#Creamos otra data de validacion
val_gen = ImageDataGenerator(validation_split=0.2, 
                                rescale=1.0/255.0,
                                width_shift_range=0.2, 
                                height_shift_range=0.2,
                                horizontal_flip=True)
val_set = val_gen.flow_from_directory(
                               directory=data_dir,
                               seed=10,
                               class_mode='sparse',
                               batch_size=batch_size,
                               shuffle=True,
                               target_size=(img_height, img_width),
                               subset='validation')

class_names = {y: x for x, y in val_set.class_indices.items()}
print(class_names)

classifier = Sequential()

classifier.add(Conv2D(32, (5, 5), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

classifier.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

classifier.build((1,img_width, img_height,3))
print(classifier.summary())

#Esta es solo la imagen original
img_tensor = np.array(pics_arr[2], dtype='int')
plt.imshow(img_tensor)
plt.show()

img_tensor = np.expand_dims(img_tensor, axis=0)
y = classifier.predict(img_tensor)
print(f"The predicted output of the sample image has a shape of {y.shape}.")

layer_outputs = [layer.output for layer in classifier.layers] 
activation_model = Model(inputs=classifier.inputs, outputs=layer_outputs) 
activations = activation_model.predict(img_tensor)

#Muestra las imagenes generadas, son muy irregulares
#Lo importante es que cada capa muestra diferentes caracterisitcas
plot_activations_multilayer(8, 8, classifier, activations)

classifier.add(Flatten()) 

classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 5, activation = 'softmax'))

classifier.summary()

classifier.compile(optimizer='adam', 
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

#El fit entrena el modelo, lo anterior fue una prediccion sin ser entrando
classifier.fit(
  train_set,
  validation_data=val_set,
  epochs=epochs
)

#Los kernel que tienen mas de una capa, se le conoce como que tienen canales
#un kernel RGB tiene 3 canales por ejemplo
for layer in classifier.layers:
    if 'conv2d' in layer.name:
        kernels, biases = layer.get_weights()
        print(f"layer name: {layer.name}, num of kernels: {kernels.shape[-1]}, kernel shape: {kernels.shape[:2]}, kernel depth: {kernels.shape[2]}")

#Grafica los valores de los kernel
for layer in classifier.layers:
    if 'conv2d' in layer.name:
        name = layer.name

        kernels, _ = layer.get_weights()
        k_min, k_max = kernels.min(), kernels.max()
        kernels = (kernels - k_min) / (k_max - k_min)

        for i in range(4):
            kernel = kernels[:,:,:,i]
            fig = plt.figure(figsize=(5, 2))
            fig.suptitle(f"{name}, kernel {i+1}", fontsize=15)

            for j in range(3):
                plt.subplot(1, 3, j+1)
                plt.imshow(kernel[:,:,j], cmap='gray')
                plt.xticks([])
                plt.yticks([])
plt.show()
                
plt.imshow(kernels[:,:,2,3], cmap='gray')
plt.show()

layer_outputs = [layer.output for layer in classifier.layers]
activation_model = Model(inputs=classifier.inputs, outputs=layer_outputs)

img_tensor = pics_arr[1]
plt.imshow(np.array(img_tensor, dtype='int'))

img_tensor = np.expand_dims(img_tensor, axis=0)

activations = activation_model.predict(img_tensor)[:6]

plot_activations_multilayer(7,8,classifier,activations)

#Predijo correctamente la clase tulipan
y = classifier.predict(img_tensor)
label = class_names[np.argmax(y)]

plt.imshow(img_tensor.reshape((img_width,img_height,3)).astype("uint8"))
plt.title(f"Predicted class is: {label}", fontsize=13)
plt.show()


#######################################################################################################################

                                      #Ejemplos Redes neuronales convolucionales

import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#Cargamos nuestra data separando en datos de entrenamiento y prueba
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Son imagenes de baja resolucion de 32 * 32 * 3 colores
print(x_train[444].shape)

#
print(y_train[444])
plt.imshow(x_train[444])
plt.show()

num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train[444])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Tiene uuna puntuacion de 0.6
model_1 = Sequential()


## 5x5 convolution with 2x2 stride and 32 filters
model_1.add(Conv2D(32, (5, 5), strides = (2,2), padding='same',
                 input_shape=x_train.shape[1:]))
model_1.add(Activation('relu'))

## Another 5x5 convolution with 2x2 stride and 32 filters
model_1.add(Conv2D(32, (5, 5), strides = (2,2)))
model_1.add(Activation('relu'))

## 2x2 max pooling reduces to 3 x 3 x 32
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Dropout(0.25))

## Flatten turns 3x3x32 into 288x1
model_1.add(Flatten())
model_1.add(Dense(512))
model_1.add(Activation('relu'))
model_1.add(Dropout(0.5))
model_1.add(Dense(num_classes))
model_1.add(Activation('softmax'))

model_1.summary()

batch_size = 32

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0005, decay=1e-6)

# Let's train the model using RMSprop
model_1.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model_1.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=15,
              validation_data=(x_test, y_test),
              shuffle=True)

###Ejemplo 2


#Mejora la puntuacion a 0.7
model_2 = Sequential()

model_2.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model_2.add(Activation('relu'))
model_2.add(Conv2D(32, (3, 3)))
model_2.add(Activation('relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Dropout(0.25))

model_2.add(Conv2D(64, (3, 3), padding='same'))
model_2.add(Activation('relu'))
model_2.add(Conv2D(64, (3, 3)))
model_2.add(Activation('relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Dropout(0.25))

model_2.add(Flatten())
model_2.add(Dense(512))
model_2.add(Activation('relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(num_classes))
model_2.add(Activation('softmax'))

model_2.summary()

# initiate RMSprop optimizer
opt_2 = keras.optimizers.RMSprop(learning_rate=0.0005)

# Let's train the model using RMSprop
model_2.compile(loss='categorical_crossentropy',
              optimizer=opt_2,
              metrics=['accuracy'])

model_2.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              validation_data=(x_test, y_test),
              shuffle=True)


#######################################################################################################################

                                                 #API en Keras

#Hay 3 formas de hacer modelos en Keras

#Modelo secuancial: El cual apila capas que le especifiquemos, tiene 1 input y 1 output

#API funcional: Puede admitir arquitecturas mas flexibles, arbitrarias y mas complejas

#Subclasificacion: Esto es para modelos inovadores que empiezen desde 0



import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
print(tf.__version__)
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split
seed = 7
np.random.seed(seed)

dataframe = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module1/L1/data/sonar.csv", header=None)

dataset = dataframe.values

X = dataset[:, 0:60].astype(float)
y = dataset[:, 60]

# encode labels
le = LabelEncoder()
encoded_y = le.fit(y).transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size=0.20, random_state=42)


#Creamos un modelos SECUENCIAL

def baseline_model():
    model = Sequential()
    model.add(Dense(60, activation='relu', input_shape=(60,)))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

estimator = baseline_model()
estimator.summary()

estimator.fit(X_train, y_train, epochs=10, batch_size=16)

y_pred = estimator.predict(X_test)
y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
metrics.accuracy_score(y_pred, y_test)


###Modelo funcional API

#Un modelo funcional es un funcion que hace todo, que es un poco diferente en su sintaxis
#El modelo secuencial generalmente se ocupa para problemas mas simples, pero en general se prefiere ocupar un
#modelo API al poder hacer arquitecturas mas complejas
#Un modelo funcional API es un grafico o esquema que permite tener multiples capas y salidas, ademas de muchas
#formas

#Todos los modelos que se pueden hacer en secuencial se pueden hacer en un modelo funcional API, inluyen usos
#como ResNet, GoogleNet, Xception

#Tanto secuencial como funcional ocupan la misma funcion de entrenamiento


def functional_model():
    inputs = keras.Input(shape=(60,))
    layer1 = Dense(60, activation='relu')(inputs)
    layer2 = Dense(60, activation='relu')(layer1)
    outputs = Dense(1, activation='sigmoid')(layer2)

    model = keras.Model(inputs, outputs)
    
    # Compile model, write code below
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

functional_estimator=functional_model()
estimator.summary()

functional_estimator.fit(X_train, y_train, epochs=10, batch_size=16)

y_pred = functional_estimator.predict(X_test)
y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
metrics.accuracy_score(y_pred, y_test)


###Modelo Subclassificacion

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(60, activation = 'relu')
        self.dense2 = Dense(60, activation = 'relu')
        self.dense3 = Dense(1, activation = 'sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)
    
def subclass_model():
    inputs = keras.Input(shape=(60,))
    mymodel = MyModel()
    outputs = mymodel.call(inputs)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    return model

subclass_estimator = subclass_model()
subclass_estimator.fit(X_train, y_train, epochs=15, batch_size=16)

y_pred = subclass_estimator.predict(X)
y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
metrics.accuracy_score(y_pred, encoded_y)


#######################################################################################################################

                                         #Aprendizaje por transferencia
                                              #Diferentes modelos

#El aprendizaje por transferencia se refiere a ocupar modelos previamete entrenados con cantidades de datos
#gigantes, donde podemos ocupar ese entrenamiento para detectar caracteristicas

import numpy as np
import datetime
import os
import random, shutil
import glob
import skillsnetwork

import warnings
warnings.simplefilter('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from matplotlib.image import imread

from os import makedirs,listdir
from shutil import copyfile
from random import seed
from random import random
import keras 
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from sklearn import metrics


sns.set_context('notebook')
sns.set_style('white')

#Un flujo de trabajo comun es:
#1. Iniciar modelo base y pre-entrenar los pesos como ImageNet
#2. COngelar las capas en la base del modelo, ajustando training=False
#3. Definir un nuevo modelo que se superponga a la salida de las capas del modelo base
#4. Entrenar modelo resultante en tu dataset

#Descargamos las imagenes

img_rows, img_cols = 150, 150
batch_size = 32
n_epochs = 10
n_classes = 2
val_split = 0.2
verbosity = 1
path = 'o-vs-r-split/train/'
path_test = 'o-vs-r-split/test/'
input_shape = (img_rows, img_cols, 3) #RGB
labels = ['O', 'R']
seed = 10
checkpoint_path='ORnet.h5'

#Generamos imagenes
train_datagen = ImageDataGenerator(
    validation_split = val_split,
    rescale=1.0/255.0,
	width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    validation_split = val_split,
    rescale=1.0/255.0,
	width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255.0
)

train_generator = train_datagen.flow_from_directory(
    directory = path,
    classes = labels,
    seed = seed,
    batch_size = batch_size, 
    class_mode='binary',
    shuffle = True,
    target_size=(img_rows, img_rows),
    subset = 'training'
)

val_generator = val_datagen.flow_from_directory(
    directory = path,
    classes = labels,
    seed = seed,
    batch_size = batch_size, 
    class_mode='binary',
    shuffle = True,
    target_size=(img_rows, img_rows),
    subset = 'validation'
)

test_generator = test_datagen.flow_from_directory(
    directory = path_test,
    classes = labels,
    class_mode='binary',
    seed = seed,
    batch_size = batch_size, 
    shuffle = True,
    target_size=(img_rows, img_rows)
)

IMG_DIM = (100, 100)

train_files = glob.glob('./o-vs-r-split/train/O/*')
train_imgs = [tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('/')[3].split('.')[0].strip() for fn in train_files]

img_id = 0
O_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1],
                                   batch_size=1)
O = [next(O_generator) for i in range(0,5)]
fig, ax = plt.subplots(1,5, figsize=(16, 6))
print('Labels:', [item[1][0] for item in O])
l = [ax[i].imshow(O[i][0][0]) for i in range(0,5)]


#Aprovechamos un modelo pre-entrenado para extraert las caracterisitcas geneticas

#Cargamos el modelo preentrenado
from keras.applications import vgg16
input_shape = (150, 150, 3)

vgg = vgg16.VGG16(include_top=False,
                        weights='imagenet',
                        input_shape=input_shape)

output = vgg.layers[-1].output
output = tf.keras.layers.Flatten()(output)
basemodel = Model(vgg.input, output)

#Creamos el entrenamiento sin entrenamiento
basemodel.trainable = False
for layer in basemodel.layers: layer.trainable = False

input_shape = basemodel.output_shape[1]

model = Sequential()
model.add(basemodel)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

#Al compilar el objeto congela las caracteristicas entrenadas
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['accuracy'])


#Utilizamos la parada rapida para evitar sobreentrenar el modelo, esto podria causar sobreajuste
#Probamos el modelo con los datos ficticios
from keras.callbacks import LearningRateScheduler
checkpoint_path='O_R_tlearn_image_augm_cnn_vgg16.keras'

# define step decay function
class LossHistory_(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(len(self.losses)))
        print('lr:', exp_decay(len(self.losses)))

def exp_decay(epoch):
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate

# learning schedule callback
loss_history_ = LossHistory_()
lrate_ = LearningRateScheduler(exp_decay)

keras_callbacks = [
      EarlyStopping(monitor = 'loss', 
                    patience = 5, 
                    mode = 'min', 
                    min_delta=0.01),
      ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, mode='min')
]

callbacks_list_ = [loss_history_, lrate_, keras_callbacks]

#Entrenamos el modelo
extract_feat_model = model.fit(train_generator, 
                              steps_per_epoch=10, 
                              epochs=5,
                              validation_data=val_generator, 
                              validation_steps=10, 
                              verbose=1,
                              callbacks = callbacks_list_)  
model.save('O_R_tlearn_image_augm_cnn_vgg16.keras')

###El ajuste fino es un paso opcional

#generalmente el ajuste fino termina mejorando el modelo, tiende a sobreajustar por lo que necesita regulacion
#le agrega un nuevo entrenamiento

[layer.name for layer in basemodel.layers]

basemodel.trainable = True

set_trainable = False

for layer in basemodel.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


###Creamos el nuevo modelo con la entrada del modelo anterior

model = Sequential()
model.add(basemodel)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

checkpoint_path='O_R_tlearn_image_augm_fine_tune_vgg16.keras'


# learning schedule callback
loss_history_ = LossHistory_()
lrate_ = LearningRateScheduler(exp_decay)

keras_callbacks = [
      EarlyStopping(monitor = 'loss', 
                    patience = 5, 
                    mode = 'min', 
                    min_delta=0.01),
      ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, mode='min')
]

callbacks_list_ = [loss_history_, lrate_, keras_callbacks]

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-5),
              metrics=['accuracy'])
              
fine_tune_model = model.fit(train_generator, 
                    steps_per_epoch=10, 
                    epochs=5,
                    callbacks = callbacks_list_,   
                    validation_data=val_generator, 
                    validation_steps=10, 
                    verbose=1)    
model.save('O_R_tlearn_image_augm_fine_tune_vgg16.keras')
extract_feat_model = tf.keras.models.load_model('O_R_tlearn_image_augm_cnn_vgg16.keras')
fine_tune_model = tf.keras.models.load_model('O_R_tlearn_image_augm_fine_tune_vgg16.keras')

from sklearn.utils import shuffle


IMG_DIM = (150, 150)

# Read in all O and R test images file paths. Shuffle and select 50 random test images. 
test_files_O = glob.glob('./o-vs-r-split/test/O/*')
test_files_R = glob.glob('./o-vs-r-split/test/R/*')
test_files = test_files_O + test_files_R
test_files = shuffle(test_files)[0:50]

# Extract label from file path
test_imgs = [tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=IMG_DIM)) for img in test_files]
test_imgs = np.array(test_imgs)
test_labels = [fn.split('/')[3].split('.')[0].strip() for fn in test_files]

# Standardize
test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255

class2num_lt = lambda l: [0 if x == 'O' else 1 for x in l]
num2class_lt = lambda l: ['O' if x < 0.5 else 'R' for x in l]

test_labels_enc = class2num_lt(test_labels)

predictions_extract_feat_model = extract_feat_model.predict(test_imgs_scaled, verbose=0)
predictions_fine_tune_model = fine_tune_model.predict(test_imgs_scaled, verbose=0)

predictions_extract_feat_model = num2class_lt(predictions_extract_feat_model)
predictions_fine_tune_model = num2class_lt(predictions_fine_tune_model)

print('Extract Features Model')
print(metrics.classification_report(test_labels, predictions_extract_feat_model))
print('Fine-Tuned Model')
print(metrics.classification_report(test_labels, predictions_fine_tune_model))

custom_im = test_imgs_scaled[3]
plt.imshow(custom_im)
plt.show()

num2class_lt(extract_feat_model.predict(custom_im.reshape((1,
                                                           test_imgs_scaled.shape[1], 
                                                           test_imgs_scaled.shape[2], 
                                                           test_imgs_scaled.shape[3])), verbose=0))


###Ejemplo 2


dataset_home = 'signs/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    labeldirs = ['stop/', 'not_stop/']
    for labeldir in labeldirs:
        newdir = dataset_home + subdir + labeldir
        makedirs(newdir, exist_ok = True)

for file in listdir("stop"):
    if file != '.DS_Store':
        shutil.copyfile(f"stop/{file}", f"signs/train/stop/{file}")
        
for file in listdir("not_stop"):
    if file != '.DS_Store':
        shutil.copyfile(f"not_stop/{file}", f"signs/train/not_stop/{file}")

test_path = "test_set_stop_not_stop/"
for file in listdir(test_path):
    if file.startswith("stop"):
        shutil.copyfile(test_path+file, f"signs/test/stop/{file}")
    elif file.startswith("not_stop"):
        shutil.copyfile(test_path+file, f"signs/test/not_stop/{file}")  


train_stop = glob.glob('./signs/train/stop/*')
train_not_stop = glob.glob('./signs/train/not_stop/*')

fig1, ax1 = plt.subplots(1,5,figsize=(15,4))
fig1.suptitle("STOP Signs", fontsize=18)
l1 = [ax1[i].imshow(imread(train_stop[i])) for i in range(5)]

fig2, ax2 = plt.subplots(1,5,figsize=(15,4))
fig2.suptitle("NO STOP Signs", fontsize=18)
l2 = [ax2[i].imshow(imread(train_not_stop[i])) for i in range(5)]


path = "signs/train/"
labels = ['stop', 'not_stop']
seed = 123
batch_size = 30
target_size = (112,112)


train_datagen = ImageDataGenerator(validation_split=0.2,
                                  rescale=1./255.,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

val_datagen = ImageDataGenerator(validation_split=0.2,
                                  rescale=1./255.)


train_generator = train_datagen.flow_from_directory(
    directory = path,
    classes = labels,
    seed = seed,
    batch_size = batch_size, 
    class_mode='binary',
    shuffle = True,
    target_size=target_size,
    subset = 'training'
)

val_generator = val_datagen.flow_from_directory(
    directory = path,
    classes = labels,
    seed = seed,
    batch_size = batch_size, 
    class_mode='binary',
    shuffle = True,
    target_size=target_size,
    subset = 'validation'
)

print(train_generator.class_indices)

prob2class = lambda x: 'Stop' if x < 0.5 else 'Not Stop' 


test_files = glob.glob('signs/test/stop/*.jpeg') + glob.glob('signs/test/not_stop/*.jpeg')
test_files = shuffle(test_files)

test_imgs = [tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=target_size)) for img in test_files]
test_imgs = np.array(test_imgs).astype('int')

# Standardize
test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255


def build_compile_fit(basemodel):
    
    # flatten the output of the base model
    x = Flatten()(basemodel.output)
    # add a fully connected layer 
    x = Dense(1024, activation='relu')(x)
    # add dropout layer for regularization
    x = Dropout(0.2)(x)
    # add final layer for classification
    x = Dense(1, activation='sigmoid')(x)

    model = Model(basemodel.input, x)
    model.compile(optimizer = optimizers.RMSprop(learning_rate=0.0001),
                                                       loss='binary_crossentropy',
                                                       metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor = 'loss', 
                    patience = 5, 
                    mode = 'min', 
                    min_delta=0.01)]

    model.fit(train_generator,
              validation_data = val_generator,
              steps_per_epoch=5, # num of batches in one epoch
              epochs=10,
              callbacks=callbacks)
    
    return model
    
    
###Modelo pre-entrenado Inception-V5

#En vez de intentar agrandar la profundidad de la red, el modelo agranda el ancho y la profundidad
#logrando mejor precision

#Se centra mas en un trabajo en simultaneo, obteniendo caracteristicas en proceso simultaneos

from keras.applications.inception_v3 import InceptionV3

# initialize the base model
basemodel = InceptionV3(input_shape=(112,112,3),
                          include_top = False,
                          weights = 'imagenet')

for layer in basemodel.layers:
    layer.trainable = False

# call the build_compile_fit function to complete model training
inception_v3 = build_compile_fit(basemodel)

fig, ax = plt.subplots(2, 4, figsize=(12, 6))

for i, ax in enumerate(ax.flat):
    ax.imshow(test_imgs[i])
    pred_class = prob2class(inception_v3.predict(test_imgs_scaled[i].reshape(1, 112, 112,3)))
    
    # print the predicted class label as the title of the image
    ax.set_title(pred_class, fontsize=15)
    ax.axis("off")
plt.show()


###Modelo pre-entrenado MobileNet

#MobileNets son arquitecturas de aprendizaje profundo pequeñas y muy eficientes especialmente diseñadas
#para dispositivos móviles

#Utiliza un nuevo tipo de capa de convolución, conocida como convolución separable en profundidad. La principal
#diferencia entre una convolución 2D y una convolución en profundidad es que la primera se realiza en múltiples
#canales de entrada haciendo una suma ponderada de los píxeles de entrada con el filtro, mientras que la segunda
#se realiza por separado en cada canal.

from keras.applications.mobilenet import MobileNet

# initialize the base model
basemodel = MobileNet(input_shape=(112,112,3),
                          include_top = False,
                          weights = 'imagenet')

for layer in basemodel.layers:
    layer.trainable = False
    
# call the build_compile_fit function to complete model training
mobile_net = build_compile_fit(basemodel)

fig, ax = plt.subplots(2, 4, figsize=(12, 6))

for i, ax in enumerate(ax.flat):
    ax.imshow(test_imgs[i])
    pred_class = prob2class(mobile_net.predict(test_imgs_scaled[i].reshape(1, 112, 112,3)))
    
    # print the predicted class label as the title of the image
    ax.set_title(pred_class, fontsize=15)
    ax.axis("off")
plt.show()
    

###Modelo pre-entrenado ResNet-50

#ResNet presenta conexiones de salto especiales que agregan la salida de una capa anterior directamente a una
#capa posterior y un uso intensivo de la normalización por lotes. Nos permite diseñar CNN profundas sin comprometer
#la convergencia y precisión del modelo. Los componentes básicos de ResNets son los bloques de convolución y de identidad.

#Básicamente, ResNet utiliza las capas de red para ajustarse a un mapeo residual, en lugar de intentar aprender el
#mapeo subyacente deseado directamente con capas apiladas

from keras.applications import ResNet50

# initialize the base model
basemodel = ResNet50(input_shape=(112,112,3),
                          include_top = False,
                          weights = 'imagenet')

for layer in basemodel.layers:
    layer.trainable = False

# call the build_compile_fit function to complete model training
resnet_50 = build_compile_fit(basemodel)

fig, ax = plt.subplots(2, 4, figsize=(12, 6))

for i, ax in enumerate(ax.flat):
    ax.imshow(test_imgs[i])
    pred_class = prob2class(resnet_50.predict(test_imgs_scaled[i].reshape(1, 112, 112,3)))
    
    # print the predicted class label as the title of the image
    ax.set_title(pred_class, fontsize=15)
    ax.axis("off")
plt.show()    


#######################################################################################################################

                                            #Tecnicas de regularizacion

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

#Funcion prepar la data, cambia nombres y elimina  columnas
def prepare_data():
    try:
        data = pd.read_csv("spam.csv", encoding='latin-1')
    except FileNotFoundError:
        print("Data file not found, make sure it's downloaded.")
        
    data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
    data.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
    data.label = data['label'].map({'ham':0, 'spam':1})
    data['Count'] = data['text'].apply(lambda x: len(x))
    
    sw=stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=sw, binary=True)

    X = vectorizer.fit_transform(data.text).toarray()
    y = data.label
    
    return X, y


def plot_metrics(history, titulo):
    fig = plt.figure(figsize=(10,5))
    for i, metric in enumerate(['accuracy', 'loss']):
        train_metrics = history.history[metric]
        val_metrics = history.history['val_'+metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.subplot(1,2,i+1)
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title('Training and validation '+ metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_"+metric, 'val_'+metric])
        plt.title(titulo)
    plt.show()


###Sobreajuste

#En un modelo de limite de decision de dos clases, un modelo puede sufrir sobreajuste, separa correctamente
#las dos clases, pero comete muchos errores
#Pero se puede hacer otro modelo que sea muy variable en su limite de decision, pero que los clasifique correctamente

#A pesar de que ese modelo lo hace correctamente, que sea tan variable para lograr clasificar esos datos, hace
#que sufra un sobreajuste porque ese modelo para otro conjunto de datos no lograra generalizar bien

#Para combatir el sobreajuste existen reguladores estos son:
#1. Lasso
#2. Ridge
#3. Dropout
#4. Batch Normalizacion
#5. Datos barajados (shuffling)


###L2 Ridge

#Regulariza el cuadrado de los pesos

tf.keras.regularizers.l2(l2=0.01) 

dense_layer = Dense(32, 
                activation="relu", 
                kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))



###L1 Lasso

#Penaliza los pesos mas grandes y las caracterisitcas menos importantes las lleva a 0

dense_layer = Dense(32, 
            activation="relu", 
            kernel_regularizer=tf.keras.regularizers.l1(l1=0.01))

dense_layer = Dense(32, 
                activation="relu", 
                kernel_regularizer="l1")


###Dropout (Abandono)

#Es una tecnica que hace que ciertas capas no ocupen todas las neuronas, en cada iteracion se ocuparan
#neuronas al azar en esa capa, esto hace que el modelo no se sobreajuste y generalize mejor.

#Esto tambien hace que los pesos se actualizen de forma no pareja, por lo que esto agregara ruido a los pesos,
#entoncces cuando el modelo intente corregir problemas en exceso, el dropout interrumpe el proceso evitando
#el sobreajuste

from tensorflow.keras.layers import Dropout

dropout_layer = Dropout(rate=0.2)


###Normalizacion por lotes (Batch Normalitation)

#La normalizacion por lotes hace que se relentize el aprendizaje cuando se creo, provocando lentitud en los
#ajuste de los pesos

#Lo que hace es normalizar cada mini lote, esto desestabiliza el proceso, logrando que el ajuste del modelo
#se redusca, recordar que la normalizacion es cuando los datos se dejan con media=0 y std=1, por lo tanto
#cada lote sigue una distribion normal

#Esta normalizaion hace que aprenda mucho mas rapido en la practica.

from tensorflow.keras.layers import Dense, BatchNormalization

batchnorm_layer = BatchNormalization()


###Ejemplo de regularizacion

#Generamos una data que sigan cierta funcion especifica
def generate_data(seed=43,std=0.1,samples=500):
    np.random.seed(seed)
    X =np.linspace(-1,1,samples)
    f = X**3 +2*X**2 -X 
    y=f+np.random.randn(samples)*std
    
    return X, y

#Plot de la funcion y los datos
X,y = generate_data()
f = X**3 +2*X**2 -X
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.title("data and true function")
plt.legend()
plt.show()

y[20:30] = 0
y[100:110] = 2
y[180:190] = 4
y[260:270] = -2
y[340:350] = -3
y[420:430] = 4

#Otro plot que muestra datos lejanos
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.title("Con datos lejanos")
plt.legend()
plt.show()


from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#Ajustamos un modelo
model = Sequential()
model.add(Dense(1000, activation='relu',input_shape=(1,)))
model.add(Dense(120,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(1))

y_pred = model.predict(X)

#Ploteo de las predicciones del modelo, antes de ser entrenado
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred ,label="predicted function")
plt.title("Antes de ser entrenado")
plt.legend()
plt.show()

model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model.fit(X, y,  epochs=20, batch_size=100)

y_pred = model.predict(X)
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred,label="predicted function")
plt.title("Modelo ajustado")
plt.legend()
plt.show()


#Calculo del error
no_reg = np.mean((y-y_pred)**2)
print(f"Mean squared error is {no_reg}\n")


#Calculo de otro modelo Lasso
model_l1 = Sequential()

model_l1.add(Dense(1000, activation='relu',input_shape=(1,),kernel_regularizer=keras.regularizers.l1(l1=0.01)))
model_l1.add(Dense(120,activation='relu',kernel_regularizer=keras.regularizers.l1(l1=0.001)))
model_l1.add(Dense(120,activation='relu'))
model_l1.add(Dense(1))
model_l1.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_l1.fit(X, y,  epochs=20, batch_size=100)

y_pred = model_l1.predict(X)
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred,label="predicted function")
plt.title("Modelo con Lasso")
plt.legend()
plt.show()

l1 = np.mean((y-y_pred)**2)
print(f"Mean squared error is {l1}\n")

#Creando Modelo con Ridge
model_l2 = Sequential()

model_l2.add(Dense(1000, activation='relu',input_shape=(1,),kernel_regularizer=keras.regularizers.l2(l2=0.0001)))
model_l2.add(Dense(120,activation='relu',kernel_regularizer=keras.regularizers.l2(l2=0.0001)))
model_l2.add(Dense(120,activation='relu',kernel_regularizer=keras.regularizers.l2(l2=0.0001)))
model_l2.add(Dense(1))
model_l2.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_l2.fit(X, y, validation_split=0.2, epochs=20, batch_size=40)

y_pred = model_l2.predict(X)
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred ,label="predicted function")
plt.title("Modelo con Ridge")
plt.legend()
plt.show()

l2 = np.mean((y-y_pred)**2)
print(f"Mean squared error is {l2}\n")

#Modelo con Dropout
model_dp = Sequential()

model_dp.add(Dense(1000, activation='relu',input_shape=(1,)))
model_dp.add(Dropout(0.1))
model_dp.add(Dense(120,activation='relu'))
model_dp.add(Dropout(0.1))
model_dp.add(Dense(120,activation='relu'))
model_dp.add(Dropout(0.1))
model_dp.add(Dense(1))
model_dp.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_dp.fit(X, y, validation_split=0.2, epochs=20, batch_size=40)

y_pred = model_dp.predict(X)
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred ,label="predicted function")
plt.title("Modelo con Dropout")
plt.legend()
plt.show()

dp = np.mean((y-y_pred)**2)
print(f"Mean squared error is {dp}\n")

#Modelo con normalizaion de Batch
model_bn = Sequential()

model_bn.add(Dense(1000, activation='relu',input_shape=(1,)))
model_bn.add(BatchNormalization())
model_bn.add(Dense(120,activation='relu'))

model_bn.add(Dense(120,activation='relu'))
model_bn.add(Dense(1))
model_bn.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_bn.fit(X, y, validation_split=0.2, epochs=20, batch_size=40)

y_pred = model_bn.predict(X)
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred ,label="predicted function")
plt.title("Modelo con Batch Normalitation")
plt.legend()
plt.show()

bn = np.mean((y-y_pred)**2)
print(f"Mean squared error is {bn}\n")

#Modelo con la data shuffle
model_sh = Sequential()

model_sh.add(Dense(1000, activation='relu',input_shape=(1,)))
model_sh.add(Dense(120,activation='relu'))
model_sh.add(Dense(120,activation='relu'))
model_sh.add(Dense(1))

model_sh.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_sh.fit(X, y, validation_split=0.2, epochs=20, batch_size=40,shuffle=True)

y_pred = model_sh.predict(X)

plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred ,label="predicted function")
plt.title("Modelo con data shuffle")
plt.legend()
plt.show()

sh = np.mean((y-y_pred)**2)
print(f"Mean squared error is {sh}\n")

names = ['No_reg','L1','L2','Drop_out','Batch_norm','Data_shuffling']
error = [no_reg, l1, l2, dp, bn, sh]

plt.figure(figsize=(8, 4))
plt.bar(names, error, width=0.6)
plt.title("Mean Squared Error", fontsize=13)
plt.show()

for i, err in enumerate(error):
    plt.text(i-0.2, err+0.1, str(round(err,3)), color='blue', va='center') 


###Ejemplo 2

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module1/L3/data/spam.csv", encoding="latin-1")
print(data.columns)

#print(prepare_data(data))

X, y = data.iloc[:,1], data.iloc[:,0]
print(X.shape, y.shape)


#Configuracion del modelo

#La funcion get_model ofrece la posibilidad de configurar las capas y cambiar su regulacion

from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()
X=LE.fit_transform(X)
y=LE.fit_transform(y)
input_dim = X
   
def get_model(reg=None, epochs=10, titulo="Nada"):
    model = Sequential()
    model.add(Dense(512, activation='relu'))
    if reg=="L1":
        model.add(Dense(256, activation='relu', kernel_regularizer="l1"))
        model.add(Dense(64, activation='relu', kernel_regularizer="l1"))
    elif reg=="L2":
        model.add(Dense(256, activation='relu', kernel_regularizer="l2"))
        model.add(Dense(64, activation='relu', kernel_regularizer="l2"))
    elif reg=="Dropout":
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
    elif reg=="BatchNorm":
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())  
    else:
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))  
 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="adam",
                 metrics=["accuracy"])
    history = model.fit(pd.DataFrame(X), pd.DataFrame(y), batch_size=64, validation_split=0.2,
              epochs=epochs)
    plot_metrics(history, titulo)

get_model(titulo="Sin Regulacion")

#Si la presicion es mayor con el modelo sin ajustar que ajustado, el modelo se sobreajusta

get_model(reg="L1", titulo="Modelo con Lasso")

#Si el modelo mejora con este regulacion, es porque evito el sobreajuste

get_model(reg="L2", titulo="Modelo con Ridge")

#Bajo un poco la precision comparado con lasso

get_model(reg="Dropout", titulo="Modelo con Dropout")


get_model(reg="BathNorm", titulo="Modelo con Lote normalizado")


###CONCLUSION: Lasso es el que tiene mejor rendimiento, logra las mejores puntuaciones con mas iteraciones
#              Dropout es el mas estable a lo largo de las iteraciones en la puntuacion, no tiene la mejor puntuacion


#######################################################################################################################

                                     #Ejemplo Aprendizaje por trasnferencia

import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

now = datetime.datetime.now

batch_size = 128
num_classes = 5
epochs = 5

img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

def train_model(model, train, test, num_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# create two datasets: one with digits below 5 and one with 5 and above
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5


feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

model = Sequential(feature_layers + classification_layers)

model.summary()

train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)

#Me dio un modelo sobreajustado supero el valor 1 en exactitud

#Congelamos las capas

for l in feature_layers:
    l.trainable = False

model.summary()

train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)

#El modelo mejoro, pero continua sobreajustado

###Ejemplo 2

feature_layers2 = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers2 = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]
model2 = Sequential(feature_layers2 + classification_layers2)
model2.summary()

train_model(model2,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)

for l in feature_layers2:
    l.trainable = False

model2.summary()

train_model(model2,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)

#No se logra mejorar los resultados por que los modelos no son ajustados, sino que se mantienen asi



#######################################################################################################################

                                        #Tipos de datos secuenciales

#Los datos en principio son iguales y no cambian su distribucion, pero no siempre es asi existen datos
#que tienen dependencias en el tiempo, las series temporales es un ejemplo de esto

#Secuencial se refiere a cuando los datos dependen de otro dato, o mas bien de una secuencia de sucesos
#es decir de llueve hoy --> el rio crece, esto en la practica se refleja como que un dato depende de otro dato
#a esto se le llama secuencia

#Los datos financieros generalmente precentan esta particularidad

import warnings
warnings.simplefilter('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
print(tf. __version__)
import skillsnetwork

import keras 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import os
import pathlib
from scipy import signal
from scipy.io import wavfile
import re
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

sns.set_context('notebook')
sns.set_style('white')

import pandas as pd

symbols = {"TOT": "Total", "XOM": "Exxon", "CVX": "Chevron",
           "COP": "ConocoPhillips", "VLO": "Valero Energy"}
template_name = ("./financial-data/{}.csv")

quotes = {}

#Los archivos que en principio estan separados los junta en una sola data llamada quotes, lo toma por las fechas
for symbol in symbols:
    data = pd.read_csv(
        template_name.format(symbol), index_col=0, parse_dates=True
    )
    quotes[symbols[symbol]] = data["open"]
quotes = pd.DataFrame(quotes)
print(quotes.head())

import matplotlib.pyplot as plt

#Grafica los datos originales
quotes.plot()
plt.ylabel("Quote value")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
_ = plt.title("Stock values over time")
plt.show()

#Eliminamos Chevron de toda la data y lo ponemos de labels, separamos en data de entrenamiento y prueba
data, target = quotes.drop(columns=["Chevron"]), quotes["Chevron"]
data_train, data_test, target_train, target_test = train_test_split(
    data, target, shuffle=True, random_state=0)

print(data)

print(target)

#Creamos un modelo de arbol de decision
regressor = DecisionTreeRegressor()

cv = ShuffleSplit(random_state=0)

regressor.fit(data_train, target_train)
target_predicted = regressor.predict(data_test)
# Affect the index of `target_predicted` to ease the plotting
target_predicted = pd.Series(target_predicted, index=target_test.index)
print(target_predicted)

from sklearn.metrics import r2_score

test_score = r2_score(target_test, target_predicted)
print(f"The R2 on this single split is: {test_score:.2f}")

target_train.plot(label="Training")
target_test.plot(label="Testing")
target_predicted.plot(label="Prediction")

#Ploteamos lo resultado de prediccion en el tiempo gracias a que es una serie
plt.ylabel("Quote value")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
_ = plt.title("Model predictions using a ShuffleSplit strategy")
plt.show()

#La prediccion es super alta, recuerda que los datos en principio no tienen nada que ver, pero lo juntamos porque
#unos dependen de otros

data_train, data_test, target_train, target_test = train_test_split(
    data, target, shuffle=False, random_state=0,
)
regressor.fit(data_train, target_train)
target_predicted = regressor.predict(data_test)
target_predicted = pd.Series(target_predicted, index=target_test.index)

test_score = r2_score(target_test, target_predicted)
print(f"The R2 on this single split is: {test_score:.2f}")

target_train.plot(label="Training")
target_test.plot(label="Testing")
target_predicted.plot(label="Prediction")

plt.ylabel("Quote value")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
_ = plt.title("Model predictions using a split without shuffling")
plt.show()

###Sin shuffling no funciona bien, hizo la gran diferencia hacer shuffling en el primero modelo

###IMPORTANTE ES QUE PRIMER MODELO ESTA MAL

#No se pueden alterar el orden de los datos temporales, es justamente lo que no podemos hacer, hicimos trampa 


#TIPOS DE DATOS SECUENCIALES

#La variedad de datas temporales es muy variadas, por lo que pueden haber de datos medicos, financieros, musicales, etc
#pero siempre se rigen por lapsos de tiempos, estos lapsos pueden ser muy variados, lapsos por meses, dias, etc
#o con lapsos de tiempos con saltos, 

###CICLICAS

seasonality = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module4/L1/sunspotarea.csv', parse_dates=['date'], index_col='date')
plt.plot(seasonality.value)
plt.show()

seasonality_trend = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module4/L1/AirPassengers.csv', parse_dates=['date'], index_col='date')
plt.plot(seasonality_trend.value)
plt.show()

#Estas series de tiempo se les llama ciclicas, porque repiten el mismo patros siempre en cierto lapso de tiempo
#Repiten cierto patron, aunque haya un aumento o disminucion progresiva

#Por esto una secuencia ciclica, puede descomponerse en varios graficos, cada grafico mostrara una caracterisitca
#distinta del grafico original ciclico

#Esta data tiene un componente ciclico con cierta tendencia
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module4/L1/a10.csv', parse_dates=['date'], index_col='date')
plt.plot(df.value)
plt.title("Ciclica")
plt.show()

#La funcion seasonal_decompose descompone a la grafica de la secuencia ciclica, en 4 graficos
#1. Muestra la grafica original
#2. Solo muestra la tendencia de la grafica original (Hacia donde apunta)
#3. Solo muestra el patron ciclico que se repite de la grafica original
#4. Muestra las pequeñas variaciones de cada ciclo en forma de residuos
result_add = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')

#El 4. grafico lo muestra de manera aditiva, esto ser refiere que es solo el residuo en la misma escala, lo que se separa
#de la tendencia 
result_add.plot()
plt.title("Ciclica Aditiva")
plt.show()


#El 4. grafico lo muestra de manera multiplicativa, es muy poco variable, es especial para diferencias de residuos muy
#pronunciadas, para que se puedan notar comparado con otros 
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')
result_mul.plot()
plt.title("Ciclica Multiplicativa")
plt.show()


#Para quitar la tendencia de la grafica, se quita de esta forma, mostrada en un solo grafico
detrended = df.value.values - result_mul.trend
plt.plot(detrended)
plt.title('Drug Sales detrended by subtracting the trend component', fontsize=16)
plt.show()


#Para quitar el patron ciclico (sensoralidad), graficamos solo el residuo y la tendencia
deseasonalized = df.value.values / result_mul.seasonal
plt.plot(deseasonalized)
plt.show()


###Manejar falta de lapsos temporales

#Hay ciertos lapsos de tiempos que no se mostraran por diferentes razones, hay formas de enfrentarlos

#Esto es util cuando existen datos con "NaN", los rellera con otros valores para poder mostrarlos

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module4/L1/a10.csv', parse_dates=['date'], index_col='date')

from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

df['value'].plot(title="Nada")
plt.show()

#Rellenado directo (Foward Fill)
#Rellena con el ultimo valor antes del NaN
df_ffill = df.ffill()
# Print the MSE between imputed value and ground truth
error = np.round(mean_squared_error(df['value'], df_ffill['value']), 2)
df_ffill['value'].plot(title='Forward Fill (MSE: ' + str(error) +")", label='Forward Fill', style=".-")
plt.show()

#Relleno hacia atras (Backward fill)
#Rellena con el valor que viene despues del NaN
df_bfill = df.bfill()
error = np.round(mean_squared_error(df['value'], df_bfill['value']), 2)
df_bfill['value'].plot(title="Backward Fill (MSE: " + str(error) +")", label='Back Fill', color='firebrick', style=".-")
plt.show()

#Interpolacion lineal
#Se asume que el grafico sigue en linea recta, se rellena con los valores que siguen la linea recta, a esto se
#le llama interpolacion lineal esto se hace tomando en cuenta todos los datos
df['rownum'] = np.arange(df.shape[0])
df_nona = df.dropna(subset = ['value'])
f = interp1d(df_nona['rownum'], df_nona['value'])
df['linear_fill'] = f(df['rownum'])
error = np.round(mean_squared_error(df['value'], df['linear_fill']), 2)
df['linear_fill'].plot(title="Linear Fill (MSE: " + str(error) +")",label='Cubic Fill', color='green', style=".-")
plt.show()


cols = ['Id', 'Entity', 'Sentiment', 'Tweet']
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module4/L1/twitter_validation.csv', names = cols, header=None)

print(df.head(2))


#Visualizamos los datos de los tips, tipos de tweets, con la columna sentimientos
fig = plt.figure(figsize=(8,6))
df.groupby(['Sentiment']).Tweet.count().sort_values().plot.barh(
    ylim=0, title= 'tweets per category')
plt.xlabel('# of occurrences', fontsize = 12)
plt.show()


def remove_punctuation(text):
    regular_punct = list(string.punctuation)
    for punc in regular_punct:
        if punc in text:
            text = text.replace(punc, ' ')
    return text.strip().lower()

df['Tweet'] = df['Tweet'].apply(remove_punctuation)

def remove_stopwords(tweet):
    en_stops = set(stopwords.words('english'))
    tweet = tweet.split()
    tweet = " ".join([word for word in tweet if not word in en_stops])  
    return tweet

df['Tweet'] = df['Tweet'].apply(remove_stopwords)


###Exploracion de Tweet se puede convertir los mensajes de dos maneras

#TOKENIZACION

#La tokenizacion es cuando un mensaje de texto se transforma en un token, los token se refiere a una especie de
#emblema, en el contexto de mensajes de texto, es simplemente tomar la o las palabras mas relevantes de dicho mensaje

df['Tweet'] = df['Tweet'].apply(word_tokenize)

#LEMANIZACION

#El lema de un mensaje de texto se refiere a cuando una grupo de palabras independiende de los tiempos verbales o
#cantidades o contextos se refieren a lo mismo, por lo tanto son tomados como solo 1 palabra, por ejemplo
#"ir" es el lema de un grupo de palabras como: "fuimos", "vamos", "fue", "voy", todas estas palabras se representan
#solo con "ir"

def lemma_wordnet(input):
    lem = WordNetLemmatizer()
    return [lem.lemmatize(w) for w in input]

df['Tweet'] = df['Tweet'].apply(lemma_wordnet)


###VECTORIZACION

#La vectorizacion lo que hace es extraer las caracterisitcas de los textos

#Lo Hacemos con CountVectorizer()

#BoW (Bag-of-Word) (Bolsa de palabras)
#Convierte el el texto por el largo en variable, luego lo pasa a vector, sin considerar la relacion entre palabras
#Puede ser una representacion muy escasa para un gran numero de datos

def combine_text(input):
    combined = ' '.join(input)
    return combined
    
#La data viene cada mensaje en una fila y en una sola tupla (por lo que es una columna) y cada palabra separada con comas
print(df["Tweet"].shape)
df['Tweet'] = df['Tweet'].apply(combine_text)
#El texto ahora se le saca la tupla y las comas de separacion a las palabras, pero sigue siendo una sola columna
print(df["Tweet"].shape)

#Recordar que CounVectorizer lo que hace es que cada palabra de cada frase las cuenta, en el mismo orden de las filas
#hace una matriz solo con numeros, las nuevas columnas son las palabras, cada fila ahora y los valores las veces que
#se repiten esas palabras en esa fila
cv = CountVectorizer(ngram_range=(1, 1))
X_train_bow = cv.fit_transform(df['Tweet'])
#Muestra solo los 10 primeros datos
print(X_train_bow[0:10,0:10].todense())
print(X_train_bow.shape)


cv = CountVectorizer(ngram_range=(1, 1))
#Lo pasa a una sola columna aunque ya no se sabe que palabra tan explicitamente sino por numero
X_train_bow = cv.fit_transform(df['Tweet'].values.tolist()) 
print(X_train_bow)
Y_train_bow = df['Sentiment']
print(Y_train_bow)

#La matriz de CountVectorizer ahora es un bigrama
cv_bbow = CountVectorizer(ngram_range=(2, 2))
X_train_bbow = cv_bbow.fit_transform(df['Tweet'])
print(X_train_bbow) 
Y_train_bbow = df['Sentiment']
print(Y_train_bbow)

#TfidfVectorizer lo que hace es similar a lo que hace CountVectorize, trasforma en una matriz y por frecuencia, pero
#la diferencia es que TfidfVectorizer pone un numero decimal, dado que lo que pone como valor es la importancia de la
#palabra tanto para una fila como para la matriz completa, esto lo hace en base a promedios, la que tiene valores
#mas alto son las palabras mas importantes
vectorizer = TfidfVectorizer(use_idf = True, ngram_range=(1, 1))
vectorizer.fit(df)
X_train_tfidf = vectorizer.fit_transform(df['Tweet'])
print(X_train_tfidf)
Y_train_tfidf = df['Sentiment'] 
print(Y_train_tfidf)


###AUDIOS DE VOZ

#Descargamos carpeta con audios

#Extraer audios de voz

data_dir="mini_speech_commands"

#Este Readme contiente los nombres de las carpetas de cada palabra
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != './data/mini_speech_commands/README.md']
print(commands)

#Esta parte extrae las rutas
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
print(filenames)

num_samples = len(filenames)
print(num_samples)

from scipy.io.wavfile import read
test_file_name = '/no/97f4c236_nohash_3.wav'

#Extraemos el audio
rate, test_file = read(data_dir+test_file_name) 

#Rate es la cantidad de datos
print(rate)

#Test file son los niveles de sonido, se grafican automaticamente al ponerlos en plot
print(test_file)

plt.plot(test_file)
plt.show()


###Convertir las ondas en matricez

from scipy.io.wavfile import read
test_file_name = '/no/97f4c236_nohash_3.wav'

rate, test_file = read(data_dir+test_file_name)

plt.plot(test_file)
plt.show()

#Esto hace que los datos se puedan leer como dataframe
test_file_tensor = tf.io.read_file(data_dir+'/no/97f4c236_nohash_3.wav')
test_audio, _ = tf.audio.decode_wav(contents=test_file_tensor)

#16000 filas y 1 columnas, esto es una serie temporal
print(test_audio.shape)

###Espectrograma

frequencies, times, spectrogram = signal.spectrogram(test_file, rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


###Ejemplo con texto

#Contiene el codigo de DNA de los humanos, solo hay 6 tipos diferentes


human_data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module4/L1/human_data.txt", sep="\t")

print(human_data.head())
human_data['class'].value_counts().sort_index().plot.bar()
plt.title("Class distribution of Human DNA")
plt.show()

###Secuencias con K-mers

#Lo que hace es pasar los palabras a minuscula
def kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

kmers_funct('GTGCCCAGGT')

#Las datas largas ahora las corta en subdiviciones de codigos de ADN
human_data['words'] = human_data.apply(lambda x: kmers_funct(x['sequence']), axis=1)

human_data = human_data.drop('sequence', axis=1)
print(human_data.head())

#Unimos las listas
human_texts = list(human_data['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])
#separate labels
y_human = human_data.iloc[:, 0].values

#43OO por 1 columna
print("length of human text seq.:", len(human_texts))
print("length of labels", y_human.shape)

#Pasamos a matriz las secuencias de ADN
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(4,4)) 
X = cv.fit_transform(human_texts)

#Separamos en datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y_human, 
                                                    test_size = 0.20, 
                                                    random_state=42)

#MultinomialNB sirve para predicir datos de texto con frecuencias, en este caso predice cual es el grupo de ADN
#solo mostrandole diferentes patrones, la precision es muy alta
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))


#######################################################################################################################

                                         #Embendding (Incrustaciones)


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import importlib
import numpy as np

import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)

import  tensorflow as tf

tf.random.set_seed(1234)
from tensorflow.keras.layers import Embedding, Dense, Flatten,Dropout
from tensorflow.keras.models import Sequential

from keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import regularizers

def display_metrics(history):

    n = len(history.history["loss"])

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(range(n), (history.history["loss"]),'r', label="Train Loss")
    ax.plot(range(n), (history.history["val_loss"]),'b', label="Validation Loss")
    ax.legend()
    ax.set_title('Loss over iterations')

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(range(n), (history.history["acc"]),'r', label="Train Acc")
    ax.plot(range(n), (history.history["val_acc"]),'b', label="Validation Acc")
    ax.legend(loc='lower right')
    ax.set_title('Accuracy over iterations')

def plot_embedding(X_embedded,start=100,stop=300,sample=10):
    fig, ax = plt.subplots()
    ax.scatter(X_embedded[start:stop:sample,0], X_embedded[start:stop:sample,1])
    for i in range(start,stop,sample):
        ax.annotate(REVERSE_LOOKUP[i+1], (X_embedded[i,0], X_embedded[i,1]))

def rotten_tomato_score(p_yx):
    return ["rotten"  if p<=0.60 else "fresh"for p in p_yx ]
        

###Ejemplo 1, se quiere predecir que peliculas son las mejores segun los criticos

#Se trabajara con texto, lo primero que se debe hacer es codificar las palabras a numeros 

samples=['I hate cats', 
         'the dog is brown and I like cats', 
         'for the']

#Hay 2 formas de codificar, la codificacion sola es una tokenizacion

#Con one-hot encoding que lo que va hacer es que cada palabra la traspasar a un numero como una clase

#Con multi-hot encoding que lo que hacer que cada fila sea una palabra ocupada, como son 3 frases, hara 3 columnas
#cada columna entonces dira si tiene esa palabra con un 1 y 0 si no la tiene


###Embedding

#Lo que hace enbedding es separar en una multiplicacion de matricez

#Vector traspuesto con las palabras (codificadas) * Matriz que configura por orden las palabras de la frase, se compone solo de 1 y 0

#[1, 10] * [10, 3]  Configura "I hate cats"


###TOKENIZADOR

#Se compone de algunas configuraciones 

#num_words: es la cantidad maxima de palabras por frases, en el caso de la frase tener mas palabras que las permitidas
#lo que hace es quedarse solo con las mas comunes

#filters: un string que contentra los caracteres filtrados de los textos, como puntos ejemplo "Hola"

#lower: infica que los textos se convierten a minusculas

#split: str, infica el separador para dividir las palabras

#Crea el tikenizador
tokenizer = Tokenizer(num_words=11)

#Ajusta las muestras para que cree un corpus
tokenizer.fit_on_texts(samples) 

#Esto dice a que numero explicitamente las frecuencias de las palabras de todo el corpus
word_counts=tokenizer.word_counts
print(word_counts)

#Creamos un multi-hot encoder
#Recordando que es una secuencia por lo que cada columna tendra un solo 1 que dice que hay una palabra
for key in tokenizer.word_counts.keys():
    
    print(key)
    print(tokenizer.texts_to_matrix([key]))

#Esta es otra forma de codificar que muestra las frases entera separadas por una "," codificadas
for sample  in samples:
    
    print(sample)
    print(tokenizer.texts_to_matrix([sample]))

modes=[ "binary", "count", "tfidf", "freq"]
for mode in modes: 
    print("mode:",mode)
    for sample  in samples:
        
        print(sample)
        print(tokenizer.texts_to_matrix([sample],mode=mode))

#En muchos casos la codificacion es redundaten, por lo que se utiliza otro metodo

#Forma mas resumida de poner las frases solo con numeros en un orden, numero de la codificacion one-hot
for sample  in samples:
    
    print(sample)
    print(tokenizer.texts_to_sequences([sample]))


#Creamos un modelo
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input

import numpy as np


model = Sequential()

input_dim=3
output_dim=1
input_length=1
#El modelo añade una capa con Embedding
model.add(Embedding(input_dim=input_dim, output_dim=output_dim,input_length=input_length, weights=np.array([[0],[1],[2]])))

model.summary()


model.get_weights()
weights=np.array([0,1,2]).reshape(-1,1)
model.set_weights([weights])
model.get_weights()

for n in range(3):
    x=np.array([[n]])
    print("input x={}".format(n))
    z=model.predict(x)
    print("output z={}".format(z.tolist()))

z = model.predict(np.array([[0],[1],[2]]))
print("different samples in the batch dimension:\n",z)
z = model.predict(np.array([[0],[1],[2]]))
print(" multiple samples in a row: \n",z)


input_dim=4
output_dim=2
input_length=1
model = Sequential()
model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length, weights=np.array([[0,0],[0,1],[1,0],[1,1]])))

model.summary()

weights=np.array([[0,0],[0,1],[1,0],[1,1]])
model.set_weights([weights])
model.get_weights()

for n in range(4):
    x=np.array([[n]])
    print("input x={}".format(n))
    z=model.predict(x)
    print("input binary={}".format(z.tolist() ))



###Secuencias de relleno (Pooling)

#Tokenizamos el texto
tokenizer = Tokenizer(num_words=12)
tokenizer.fit_on_texts(samples) 
tokens=tokenizer.texts_to_sequences(samples)
print("tokens",tokens)

#Rellenamos las secuencias, ejemplo por ejemplo si [1,3] es una secuencia codificada, [1,2,6,7]
#La secuencia sera rellenada como [[0,0,1,3], [1,2,6,7]], por defecto rellena con 0
#value="Es el valor que rellenara"
#maxlen="valor maximo de columnas"
#padding="post" significa que el relleno es al final de las filas, no al principio

maxlen=9
x =pad_sequences(tokens, maxlen=maxlen,value=0)
print(x)

maxlen=9
x =pad_sequences(tokens, maxlen=maxlen,padding="post")
print(x)

maxlen=5
x =pad_sequences(tokens, maxlen=maxlen)
print(x)


###Ejemplo 2

#Se tienen reseñas postiivas y negativas, con y=0 y y=1 respectivamente

from keras.datasets import imdb

max_features = 10000

# change the default parameter of np to allow_pickle=True
np.load.__defaults__=(None, True, True, 'ASCII')
importlib.reload(np)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features )

for i,x in enumerate(x_train[0:3]):
    print("Sequence:",i) 
    print(x)

word_index = imdb.get_word_index()

REVERSE_LOOKUP={value:key for key, value in word_index.items()}

def get_review(x):
     return' '.join([REVERSE_LOOKUP[index ] for index in x])

get_review(x_train[0])

get_review(x_train[1])

for i,x in enumerate(x_train[0:3]):
    print("length {} of sample {}:".format(i,len(x)))

maxlen=20
x_train =pad_sequences(x_train, maxlen=maxlen)
x_test =pad_sequences(x_test, maxlen=maxlen)

print(x_test.shape)
 
 
#Aca se crea la capa de incrustacion
model = Sequential()
model.add(Embedding(10000, 8, input_length=20))
model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10,batch_size=30,validation_split=0.2)

display_metrics(history)

p_yx=model.predict(x_test[0:10])

print(rotten_tomato_score(p_yx))

###Visualizando los pesos

weights=model.layers[0].get_weights()[0]

print(weights.shape)


#TSNE reduce los componentes a 2
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(weights)

start=1
stop=600
sample=10
plot_embedding(X_embedded,start,stop,sample)

start=1
stop=100
sample=1
plot_embedding(X_embedded,start,stop,sample)


#Modelo que recrea el modelo anterior, este tiene mejor puntaje
model = Sequential()
model.add(Embedding(10000, 8, input_length=20))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10,batch_size=30,validation_split=0.2)

display_metrics(history)


#Modelo que tiene regulacion L2 Ridge
model = Sequential()
model.add(Embedding(max_features, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(500, kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.L2(l2=5e-3)))
model.add(Dropout(.4))

model.add(Dense(250, kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.L2(l2=5e-3)))
model.add(Dropout(.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
model.summary()

history = model.fit(x_train, y_train, epochs=10,batch_size=64, validation_split=0.2)
display_metrics(history)


#######################################################################################################################

                                           #Red Neuronal Recurrente (RNN)

#Una red neuronal recurrente es una red que utiliza datos secueciales de entrada
#La red neuronal clasifca no esta hecha para trabajar con datos secuenciales, para eso esta la red neuronal recurrente


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
print(tf. __version__)
import skillsnetwork
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding,Masking,LSTM, GRU, Conv1D, Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SimpleRNN
from tensorflow.keras.datasets import reuters
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
import tensorflow_hub as hub


# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_style('white')
np.random.seed(2024)

def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred)
    model_precision, model_recall, model_f1,_ = precision_recall_fscore_support(y_true, y_pred,average="weighted")
    model_results = {"accuracy":model_accuracy,
                     "precision":model_precision,
                     "recall" :model_recall,
                     "f1":model_f1}
    return model_results


###TIPOS DE REDES NEURONALES RECURRENTES (RNN)

#1. Muchos a muchos: tiene 1 secuencia de input y 1 secuencia de salida ---> Solucion partes de un discurso

#2. Muchos a uno: tiene 1 secuencia de input y valor de secuencia de salida ---> Soluciona Classificacion de texto

#3. Uno a muchos: 1 valor singular de secuencia y 1 secuencia de salida ---> Dada una imagen, predice la secuencia de imagen

#Descargamos una data

train_df = pd.read_csv("train.csv")
# shuffle the dataset 
train_df_shuffled = train_df.sample(frac=1, random_state=42)
print(train_df_shuffled.head())

X_train, X_test, y_train, y_test = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                    train_df_shuffled["target"].to_numpy(),
                                                    test_size = 0.1,
                                                    random_state=42)
print(X_train.shape, y_train.shape)

print(X_train[0:5])

text_vectorizer = TextVectorization(max_tokens=None, 
                                    #remove punctuation and make letters lowercase
                                    standardize="lower_and_strip_punctuation", 
                                    #whitespace delimiter
                                    split="whitespace", 
                                    #dont group anything, every token alone
                                    ngrams = None, 
                                    output_mode ="int",
                                    #length of each sentence == length of largest sentence
                                    output_sequence_length=None
                                    )

# number of words in the vocabulary 
max_vocab_length = 10000
# tweet average length
max_length = 15 

embedding = layers.Embedding(input_dim= max_vocab_length,
                             output_dim=128,
                             input_length=max_length)

#Obtenemos un modelo preentenado
#import tensorflow_hub as hub
#import os

#os.environ["TFHUB_CACHE_DIR"] = "/tmp/model"

#module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

#local_path = "/local/path/directory"
#tf.saved_model.save(embed, local_path)

#offline_module = hub.KerasLayer(local_path, trainable=True)
encoder_layer = hub.KerasLayer(embed,
                               input_shape=[],
                               dtype = tf.string,
                               trainable=False,
                               name="pretrained")

model = Sequential([encoder_layer, Dense(1,activation="sigmoid")], name="model_pretrained")
model.compile(loss="binary_crossentropy",
                     optimizer="adam",
                     metrics=["accuracy"])

model.fit(x=X_train,
              y=y_train,
              epochs=20,
              validation_data=(X_test,y_test))

print(calculate_results(y_true=y_test,
                  y_pred=tf.squeeze(tf.round(model.predict(X_test)))))
#Salio un error en el modelo


#######################################################################################################################

                                                #RNN, LSTM y GRU
                                             #Modelo de RNN cerrado

#Las redes neuronales recurrentes funcionan bien con modelos secuenciales a corto plazo, pero a largo plazo sufren
#problemas, esto se debe a que pierden cierta informacion a medida que pasan las capas
#Las RNN deben olvidar y esto se puede lograr con diferentes metodos como LSTM y GRU

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
print(tf. __version__)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding,Masking,LSTM, GRU, Conv1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SimpleRNN
from tensorflow.keras.datasets import reuters
from keras.utils import pad_sequences


sns.set_context('notebook')
sns.set_style('white')
np.random.seed(2024)


###LSTM

#LSTM tiene una memoria a corto plazo, se basa en que tiene 2 estados, el estado oculto y el estado de celda
#LSTM reconoce la informacion que es importante y se debe guardar a largo plazo y la menos relvante y que se debe
#eliminar
#Toda la informacion para por unas celdas de estado, en esta celda se decide donde va la informacion por las llamadas
#puertas, hay una puerta de entrada y otra de salida que eliimina informacion

x = np.linspace(0, 50, 501)
y = np.sin(x)
plt.plot(x, y)
plt.show()

df = pd.DataFrame(data=y, index=x, columns=['Sine'])

# percentage of data used for testing
test_percent = 0.1
# number of features
n_features = 1
# sequence length
length = 50
# batch size 
batch_size = 1

test_point = np.round(len(df)*test_percent)
test_ind = int(len(df)-test_point)

train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
len(generator)

model = Sequential()

model.add(LSTM(50, input_shape=(length, n_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(generator, epochs=6)

forecast = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(25):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

forecast = scaler.inverse_transform(forecast)

forecast_index = np.arange(50.1, 52.6, step=0.1)

plt.plot(df.index, df['Sine'])
plt.plot(forecast_index, forecast)
plt.show()

###GRU

#Es una version simplificada de LSTM es mucho mas corto, pasa una puerta en otra puerta, lo hace es definir si
#ira hacia puerta de entrada o de reinicio de la celda anterior, lo que simplifica mucho el proceso, no tiene puerta
#de salida. Esto lo hace mucho mas facil de entrenar que LSTM.

num_words = 10000
maxlen = 1000
test_split = 0.3

(X_train, y_train),(X_test, y_test) = reuters.load_data(num_words=num_words, test_split=0.3)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


y_train = tf.keras.utils.to_categorical(y_train, 46)
y_test = tf.keras.utils.to_categorical(y_test, 46)

model = Sequential()
model.add(Embedding(input_dim = num_words, output_dim = 300,input_length=1000))
model.add(GRU(128, dropout=0.2))
model.add(Dense(46, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train,batch_size=256,epochs=10,validation_split=0.2)
print(model.evaluate(X_test,y_test))


#######################################################################################################################

                                                #Ejemplos RNN
                                                 
from tensorflow import keras
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import SimpleRNN
from keras.datasets import imdb
from keras import initializers                                                 
                                                 
max_features = 20000  # This is used in loading the data, picks the most common (max_features) words
maxlen = 30  # maximum length of a sequence - truncate after this
batch_size = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print(x_train[123,:])  #Here's what an example sequence looks like

#SimpleRNN es un modelo preentrenado que le llaman vainilla
## Let's build a RNN

rnn_hidden_dim = 5
word_embedding_dim = 50
model_rnn = Sequential()
model_rnn.add(Embedding(max_features, word_embedding_dim))  #This layer takes each integer in the sequence and embeds it in a 50-dimensional vector
model_rnn.add(SimpleRNN(rnn_hidden_dim,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=x_train.shape[1:]))

model_rnn.add(Dense(1, activation='sigmoid'))

model_rnn.summary()

rmsprop = keras.optimizers.RMSprop(learning_rate = .0001)

model_rnn.compile(loss='binary_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model_rnn.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test))

score, acc = model_rnn.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

max_features = 20000  # This is used in loading the data, picks the most common (max_features) words
maxlen = 80  # maximum length of a sequence - truncate after this

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

rnn_hidden_dim = 5
word_embedding_dim = 50
model_rnn = Sequential()
model_rnn.add(Embedding(max_features, word_embedding_dim))  #This layer takes each integer in the sequence
model_rnn.add(SimpleRNN(rnn_hidden_dim,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=x_train.shape[1:]))

model_rnn.add(Dense(1, activation='sigmoid'))

rmsprop = keras.optimizers.RMSprop(learning_rate = .0001)

model_rnn.compile(loss='binary_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model_rnn.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test))


#Reducir lo hiperparametros mejoro el modelo
max_features = 5000  # This is used in loading the data, picks the most common (max_features) words
maxlen = 80  # maximum length of a sequence - truncate after this

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
          
rnn_hidden_dim = 5
word_embedding_dim = 20
model_rnn = Sequential()
model_rnn.add(Embedding(max_features, word_embedding_dim))  #This layer takes each integer in the sequence
model_rnn.add(SimpleRNN(rnn_hidden_dim,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=x_train.shape[1:]))

model_rnn.add(Dense(1, activation='sigmoid'))          
          
rmsprop = keras.optimizers.RMSprop(learning_rate = .0001)

model_rnn.compile(loss='binary_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])          
          
model_rnn.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test))

# Out of curiosity, run for 10 more epochs
#Lo tomo como iteraciones adicionales, mejoro la puntuacion
model_rnn.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test))


#######################################################################################################################

                                         #Codificadores automaticos
                                            #Todos los tipos

#Hay veces que al trabajar con imagenes, vamos a querer codificar-comprimir-decodificar
#Esto sirve para trabajar con imagenes comprimidas teniendo las mismas caracterisitcas
#Un codificador automatico, es un modelo de red neuronal que minimiza la diferencia entre la entrada y la salida

#La codificacion ---> tiene una entrada y una salida
#compresion ---> Es solo un mensaje (codigo) que se comprime
#La decodificacion ---> tiene entrada comprimida y una salida lista para ocupar

import os
import copy
import skillsnetwork

import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import keras
from keras import layers,Input,Sequential 
from keras.layers import Dense,Flatten,Reshape,Conv2DTranspose,Conv2D
from keras.models import Model


from keras.layers import Conv2D
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

#Algunas funciones utiles

#Ploteo de imagenes
def plot_images(top,bottom,start=0,stop=5,reshape_x=(28,28),reshape_xhat=(28,28)):
    
    '''
    this function plots images from the start index to the stop index from two datasets
    
    '''

    n_samples=stop-start

    for i,img_index in enumerate(range(start,stop)):
        
        # Display original
        ax = plt.subplot(2, n_samples, i + 1)
        plt.imshow(top[img_index].reshape(reshape_x[0], reshape_x[1]), cmap="gray")

        if i==n_samples//2:
            plt.title("original images")

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(bottom[img_index].reshape(reshape_xhat[0], reshape_xhat[1]), cmap="gray")


        if i==n_samples//2:
            plt.title("encoded images")

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
     
#Muestra la historia de una modelo
def graph_history(history, title='Log Loss and Accuracy over iterations'):

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title)
    N_plots=len(history.history.keys())
    color_list=['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w','bx','rx']
    for i,(key, items) in enumerate(history.history.items()):
        ax = fig.add_subplot(1, N_plots, i+1)
        ax.plot(items,c=color_list[i])
        ax.grid(True)
        ax.set(xlabel='iterations', title=key)
    plt.show()
    
#Añadir ruido a los datos de entrenamiento y prueba
def add_noise(x_train, x_test, noise_factor = 0.3):
    '''
    this function adds random values from a normal distribution as noises to the data 
    
    returns the noisy datasets 
    
    '''
    noise_factor = 0.3
    x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

    x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.).numpy()
    x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.).numpy()
    
    return x_train_noisy,x_test_noisy

#Plotear el codigo
def plot_code(h, y, numbers=[0,1,2,3,4,5,6,7,8,9], scale=[1,1,1]):
  
    h=h.numpy()
    color_list=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink','darkorange','lime']
    logic_array =np.zeros(len(y), dtype=bool)
    
    fig=plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    for num, color in zip(numbers, color_list):
        logic_array = (y==num)
        plt.scatter(scale[0]*h[logic_array,0],scale[1]*h[logic_array,1],scale[2]*h[logic_array,2],c=color, label=num)
 
    plt.title("3D output of encoder, colored by digits")
    plt.legend(loc=[1.1,0.3])
    plt.show()

def avg(shape, dtype=None):
    grad = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
        ]).reshape((3, 3, 1, 1))/9
    
    assert grad.shape == shape
    return keras.backend.variable(grad, dtype='float32')

a_conv = Conv2D(filters=1,
                       kernel_size=3,
                       kernel_initializer=avg,
                       strides=1,
                       padding='same')

def display_auto(Xiter,n=1,B=1):

    for b in range(B):    
        x = next(Xiter)
    
        plt.imshow(x[1].numpy()[n,:,:,0],cmap="gray")
        plt.title("input")
        plt.show()
        plt.imshow(x[0].numpy()[n,:,:,0],cmap="gray")
        plt.title("output")
        plt.show()

#Descargamos la data mnist y separamos datos de entrenamiento y prueba
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) =keras.datasets.mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

###Hay dos formar de hacer codificadores automaticos con API funcional y con subclasificador

###Forma de API funcional

input_img=Input(shape=(784,))

#Creamos un codificador
encoding_dim=36
encoder = Dense(encoding_dim, activation='relu')(input_img)

#Creamos el decodificador
decoder=Dense(784,activation='sigmoid')(encoder)

#Combinamos el codificador y el decodificador con esto
autoencoder =Model(input_img, decoder)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history=autoencoder.fit(x_train, x_train,
                epochs=25,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

graph_history(history, title='Log Loss and Accuracy over iterations')

#Vemos las predicciones que hace
xhat=autoencoder.predict(x_test)

#Ploteamos las imagenes de entrada y las de salida de la compresion
#Vemos que son bastante parecidas, mantienen su escencia y calidad en gran medida, pero no iguales
plot_images(x_test,xhat,start=0,stop=5)


###Autocodifcadores con Modelo de Subclasificacion

#En particular este metodo sirve porque se puede reutilizar

class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = Dense(latent_dim, activation='relu')
        self.decoder = Dense(784, activation='sigmoid')

    def call(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#Algunas configuraciones
#lateng_sim: Es el tamaño de la entrada

#La codificacion viene dada por
#encoder = tf.keras.Sequential([layers.Dense(latent_dim, activation='relu')])

#La decodificacion dada por
#decoder = tf.keras.Sequential([layers.Dense(784, activation='sigmoid')])

#Este es el unico parametro que necesitamos especificar
encoding_dim=36
autoencoder = Autoencoder(encoding_dim)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=autoencoder.fit(x_train, x_train,epochs=25,batch_size=256,shuffle=True,validation_data=(x_test, x_test))
graph_history(history, title='Log Loss and Accuracy over iterations')

xhat=autoencoder.predict(x_test)

#Mostramos las predicciones, se parecen bastante a las anteriores aunque 1 se ve bastante disminuida
plot_images(x_test,xhat,start=100,stop=105)

h=autoencoder.encoder(x_test)
print(h.shape)

#Esta parte muestra como estan los datos cuando estan codificados
plot_images(x_test, h.numpy(),start=200,stop=205,reshape_x=(28,28),reshape_xhat=(6,6))

#En general API funcional es mas compatible con diferentes modelos


###Ejemplo 1

#Calculamos la perdida entre cada una de las muestras
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

loss=[bce(x, x_s).numpy() for x, x_s in zip (x_test,xhat)]

indexs=np.flip(np.argsort(loss))

#Grafica la informacion que pierde cada imagen
plt.figure(figsize=(18,3))
for i, index in enumerate(indexs[0:10]):

    plt.subplot(1, 10, i+1)
    plt.imshow(x_test[index].reshape(28,28))
    plt.title(f"No.{index}")
    plt.axis("off")
plt.show()


###Eliminacion de ruido

#Podemos entrenar la red con imagenes con ruido primero

#Le agregamos ruido a las imagenes
x_train_noisy,x_test_noisy= add_noise(x_train, x_test,noise_factor = 0.4)

fig1 = plt.figure(figsize=(10,2))
fig1.suptitle("original images")
fig2 = plt.figure(figsize=(10,2))
fig2.suptitle("noisy images")

#Ploteamos las imagenes originales y las con ruido
for i, img_index in enumerate(range(5)):
    ax1 = fig1.add_subplot(1, 5, i+1)
    ax1.imshow(x_train[img_index].reshape((28,28)))
    ax1.axis("off")
    ax2 = fig2.add_subplot(1, 5, i+1)
    ax2.imshow(x_train_noisy[img_index].reshape((28,28)))
    ax2.axis("off")
plt.show()


#Una forma de eliminar ruido tambien con facilidad es aumentando la imagen de tamaño
encoding_dim=2*x_test.shape[1]
autoencoder = Autoencoder(encoding_dim)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=autoencoder.fit(x_train_noisy , x_train,epochs=25,batch_size=256,shuffle=True,validation_data=(x_test_noisy, x_test))
graph_history(history, title='Log Loss and Accuracy over iterations')
xhat=autoencoder.predict(x_test)

fig1 = plt.figure(figsize=(10,2))
fig1.suptitle("noisy images")
fig2 = plt.figure(figsize=(10,2))
fig2.suptitle("de-noised images")

for i, img_index in enumerate(range(5)):
    ax1 = fig1.add_subplot(1, 5, i+1)
    ax1.imshow(x_test_noisy[img_index].reshape((28,28)))
    ax1.axis("off")
    ax2 = fig2.add_subplot(1, 5, i+1)
    ax2.imshow(xhat[img_index].reshape((28,28)))
    ax2.axis("off")
plt.show()    
    

###Ejemplo 2

#Muestra los valores que contiene los pixeles graficados
autoencoder = Autoencoder(3)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
h=autoencoder.encoder(x_test)

plot_code(h,y_test)


###Ejemplo 3

#Muestra el resultado de una autocodificicion de las imagenes de moda, las vuelve como objeto bastante borrosa
(x_train, y_train), (x_test,y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape)

x_temp=layers.Flatten()(x_train)
x_temp_test=layers.Flatten()(x_test)
print(x_temp.shape, x_temp_test.shape)

encoding_dim=3
autoencoder = Autoencoder(encoding_dim)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=autoencoder.fit(x_temp,x_temp,epochs=25,batch_size=256,shuffle=True,validation_data=(x_temp_test,x_temp_test))

xhat=autoencoder.predict(x_temp_test)
plot_images(x_test,xhat,start=0,stop=5)



###AUTOCODIFICADORES PROFUNDOS (DEEP AUTOCODERS)

#En el ejemplo anterior se vio que los autocidicadores no siempre funncionan bien, de hecho el ultimo ejemplo es casi
#irreconocible el elemento

#Por esta razon se pueden entrenar modelos para que puedan obtener las mejores caracterisiticas

#Una de las cosas mas importantes es que las deep autocoders NO TIENEN LABELS

class Deep_Autoencoder (Model):
    def __init__(self, latent_dim_1, latent_dim_2):
        super(Deep_Autoencoder, self).__init__()
        self.latent_dim_1= latent_dim_1  
        self.latent_dim_1= latent_dim_2 
        self.encoder = Sequential([layers.Flatten(),Dense(latent_dim_1, activation='relu'),Dense(latent_dim_2, activation='relu')])
        self.decoder = tf.keras.Sequential([Dense(latent_dim_1, activation='relu'), Dense(784, activation='sigmoid'), Reshape((28, 28))])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#De forma de API:
#encoder = Sequential([layers.Flatten(),Dense(latent_dim_1, activation='relu'),Dense(latent_dim_2, activation='relu')])
#decoder = tf.keras.Sequential([ Dense(latent_dim_1, activation='relu'),Dense(784, activation='sigmoid'),Reshape((28, 28))])

latent_dim_1 =128
latent_dim_2=3
deep_autoencoder=Deep_Autoencoder(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2)


deep_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=deep_autoencoder.fit(x_train,x_train,epochs=50,batch_size=256,shuffle=True,validation_data=(x_test,x_test))

xhat=deep_autoencoder.predict(x_test)
plot_images(x_test,xhat,start=0,stop=5)

###Ejemplo de deep autocodificadores

#Descargamos la data

img_height=50
img_width=50
batch_size=100
data_dir_face=os.path.join(os.getcwd(), 'face_data')

Xface = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_face,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
    color_mode="grayscale")

X_face_copy=copy.copy(Xface)

def change_inputs(images, labels):
  
    return images, images

X_face_1=Xface.map(change_inputs)

X_iter=iter(X_face_1)

images1, images2 = next(X_iter)

print("images1 shape {}, and images2 {}".format(images1.shape, images2.shape))

display_auto(X_iter,n=1,B=1)

normalization_layer = tf.keras.layers.Rescaling(1./255)

#def blur_image(images, labels):
    
#    x = normalization_layer(images)
#    x_b=a_conv(x)
#    return x_b, x


Xface=Xface.map(lambda images, labels: (a_conv(normalization_layer(images)), labels))

Xiter=iter(Xface)
display_auto(Xiter,n=1,B=1)


###Autocodificadores convolucionales

#La diferencias con los deep autocoders es que los autodificadores convolucionales pueden, elegir mejor las caracterisitcas
#vimos que a pesar de que mejoro en el ejemplo anterior no es tan parecida a la imagen

class CNN_Autoencoder(Model):
    def __init__(self):
        super(CNN_Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(50, 50, 1)),
            Conv2D(16, (3, 3), activation='relu', padding='same', strides=1),
            Conv2D(8, (3, 3), activation='relu', padding='same', strides=1)])

        self.decoder = tf.keras.Sequential([
            Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#En forma de API funcional:

#encoder = tf.keras.Sequential([layers.Input(shape=(50, 50, 1)),Conv2D(16, (3, 3), activation='relu', padding='same', strides=1),Conv2D(8, (3, 3), activation='relu', padding='same', strides=1)])
#decoder = tf.keras.Sequential([Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'),Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'),Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

cnn_autoencoder_face=CNN_Autoencoder()

cnn_autoencoder_face.compile(optimizer='adam',  loss='mse')
history=cnn_autoencoder_face.fit(Xface,epochs=10)

graph_history(history, title='Log Loss and Accuracy over iterations')

x = next(Xiter)
Xhat=cnn_autoencoder_face.predict(x[0])

for image_b, image_db in zip(x[0].numpy()[0:5,:,:,0],Xhat[0:5,:,:,0]):
    plt.imshow(image_b, cmap="gray")
    plt.title("blurred image")
    plt.show()
    plt.imshow(image_db, cmap="gray")
    plt.title("de-blurred image")
    plt.show()


#######################################################################################################################

                                           #Ejemplo autocodificadores
                                    #Termine con un error en el ejemplo de VAE
                 
import warnings
warnings.simplefilter("ignore")

import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
from keras.datasets import mnist
import numpy as np
np.set_printoptions(precision=2)

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float') / 255.
x_test = x_test.astype('float') / 255.

#Reducimos las imagenes a una sola columna
x_train_flat = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_flat = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train_flat.shape)
print(x_test_flat.shape)

#Escalamos los x de entrenamiento
from sklearn.preprocessing import StandardScaler
s = StandardScaler().fit(x_train_flat)
x_train_scaled = s.transform(x_train_flat)

#Hacenos una funcion con PCA, donde se ajusta y se transofrma, primer resultado es ajustado y segundo transformado
from sklearn.decomposition import PCA

def mnist_pca(x_data, n_components):
    pca = PCA(n_components=n_components)

    fit_pca = pca.fit(x_data)
    
    print("Variance explained with {0} components:".format(n_components), 
          round(sum(fit_pca.explained_variance_ratio_), 2))

    return fit_pca, fit_pca.transform(x_data)

#Tenemos los datos ajustados, y transformados PCA, obtenemos 874 caracterisitcas que es la cantidad de columnas
pca_full, mnist_data_full = mnist_pca(x_train_scaled, 784)    

#Graficamos la explicacion con cada caracterisitca de PCA
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.title("Proportion of PCA variance\nexplained by number of components")
plt.xlabel("Number of components")
plt.ylabel("Proportion of variance explained")
plt.show()    

#Ahora solo tomamos 2 y las graficamos
pca_2, mnist_data_2 = mnist_pca(x_train_scaled, 2)    

#Graficamos los 2 componentes
num_images_per_class = 250
fig = plt.figure(figsize=(12,12))
for number in list(range(10)):
    mask = y_train == number
    x_data = mnist_data_2[mask, 0][:num_images_per_class]
    y_data = mnist_data_2[mask, 1][:num_images_per_class]
    plt.scatter(x_data, y_data, label=number, alpha=1)
plt.legend() 
plt.show()

#Ahora tomamos 64 caracteristicas, escala denuevo el mismo valor
pca_64, mnist_data_64 = mnist_pca(x_train_scaled, 64)
s = StandardScaler().fit(x_test_flat)
x_test_scaled = s.transform(x_test_flat)

#El pca_64 era solo el fit ahora lo trasforma y le aplica inversa
x_test_flat_64 = pca_64.transform(x_test_scaled)
x_test_reconstructed_64 = pca_64.inverse_transform(x_test_flat_64)

print(x_test_reconstructed_64.shape)

#true es los data de prueba escalados, y reconstruted los datos de prueba con inversa
true = x_test_scaled
reconstructed = x_test_reconstructed_64

#Lo que hace es sacar el error cuadrado medio, power eleva todo a 2 divide por la cantidad de columnas
def mse_reconstruction(true, reconstructed):
    return np.sum(np.power(true - reconstructed, 2) / true.shape[1])

#La diferencia de los datos de prueba escalados con los escalados inversos, es alta la diferencia
print(mse_reconstruction(true, reconstructed))


###Creamos un modelo simple de autocodificacion, simple porque solo tiene codificacion y decodificacion
from keras.layers import Input, Dense
from keras.models import Model

ENCODING_DIM = 64

# Encoder model
inputs = Input(shape=(784,)) 
encoded = Dense(ENCODING_DIM, activation="sigmoid")(inputs)
encoder_model = Model(inputs, encoded, name='encoder')

# Decoder model
encoded_inputs = Input(shape=(ENCODING_DIM,), name='encoding')
reconstruction = Dense(784, activation="sigmoid")(encoded_inputs)
decoder_model = Model(encoded_inputs, reconstruction, name='decoder')

# Defining the full model as the combination of the two
outputs = decoder_model(encoder_model(inputs))
full_model = Model(inputs, outputs, name='full_ae')

full_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

#Ajustamos los datos a la autocoficacion con deep learining
history = full_model.fit(x_train_flat, x_train_flat, shuffle=True, epochs=1, batch_size=32)

#Tenemos las imagnenes ahora reducidas en calidad
encoded_images = encoder_model.predict(x_test_flat)
print(encoded_images.shape)

#Valores de una sola imagen
print(encoded_images[0])

#Ahora calculamos el error con autocodificacion con deep learning
decoded_images = full_model.predict(x_test_flat)
print(mse_reconstruction(decoded_images, x_test_flat))


#Creamos otro modelo con una segunda capa neuronal, con activacion sigmoid, tanto para la codificacion como la
#decodificacion

ENCODING_DIM = 64
HIDDEN_DIM = 256
### BEGIN SOLUTION
# Encoder model
inputs = Input(shape=(784,)) 
encoded = Dense(ENCODING_DIM, activation="relu")(inputs)
encoder_hidden = Dense(HIDDEN_DIM, activation="sigmoid")(encoded)
encoder_model = Model(inputs, encoded, name='encoder')

# Decoder model
encoded_inputs = Input(shape=(ENCODING_DIM,), name='encoding')
decoder_hidden = Dense(HIDDEN_DIM, activation="relu")(encoded_inputs)
reconstruction = Dense(784, activation="sigmoid")(decoder_hidden)
decoder_model = Model(encoded_inputs, reconstruction, name='decoder')

# Defining the full model as the combination of the two
outputs = decoder_model(encoder_model(inputs))
full_model = Model(inputs, outputs, name='full_ae')

full_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

history = full_model.fit(x_train_flat, x_train_flat, shuffle=True, epochs=2, batch_size=32)

decoded_images = full_model.predict(x_test_flat)
print(mse_reconstruction(decoded_images, x_test_flat))


###Ejemplo 3

#Hasta ahora solo habiamos ocupado solo 1 epoca, veamos aplicandole mas

def train_ae_epochs(num_epochs=1):
### BEGIN SOLUTION
    ENCODING_DIM = 64
    HIDDEN_DIM = 256

    # Encoder model
    inputs = Input(shape=(784,)) 
    encoded = Dense(ENCODING_DIM, activation="relu")(inputs)
    encoder_hidden = Dense(HIDDEN_DIM, activation="sigmoid")(encoded)
    encoder_model = Model(inputs, encoded, name='encoder')

    # Decoder model
    encoded_inputs = Input(shape=(ENCODING_DIM,), name='encoding')
    decoder_hidden = Dense(HIDDEN_DIM, activation="relu")(encoded_inputs)
    reconstruction = Dense(784, activation="sigmoid")(decoder_hidden)
    decoder_model = Model(encoded_inputs, reconstruction, name='decoder')

    # Defining the full model as the combination of the two
    outputs = decoder_model(encoder_model(inputs))
    full_model = Model(inputs, outputs, name='full_ae')
    
    full_model = Model(inputs=inputs, 
                       outputs=outputs)

    full_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    mse_res = []
    for i in range(num_epochs):
        history = full_model.fit(x_train_flat, x_train_flat, shuffle=True, epochs=1, batch_size=32)
    
        decoded_images = full_model.predict(x_test_flat)
        reconstruction_loss = mse_reconstruction(decoded_images, x_test_flat)
        mse_res.append(reconstruction_loss)
        print("Reconstruction loss after epoch {0} is {1}"
              .format(i+1, reconstruction_loss))
### END SOLUTION       
    return mse_res

print(train_ae_epochs(5))


#En cada epoca se reduce aun mmas el error


###Codificador automatica tradicional (VAE)

#Un VAE tiene pasos que se rigen por la distribucion estandar y la desviacion estandar

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from tensorflow.keras.losses import MSE, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import os

def sampling(args):
    
    #Transforms parameters defining the latent space into a normal distribution.
    
    # Need to unpack arguments like this because of the way the Keras "Lambda" function works.
    mu, log_sigma = args
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=tf.shape(mu))
    sigma = K.exp(log_sigma)
    return mu + K.exp(0.5 * sigma) * epsilon

hidden_dim = 256
batch_size = 128
latent_dim = 2 
# this is the dimension of each of the vectors representing the two parameters
# that will get transformed into a normal distribution
epochs = 1


# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(784, ), name='encoder_input')
x = Dense(hidden_dim, activation='relu')(inputs)


z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
# NOTE: output of encoder model is *2* n-dimensional vectors:

z = Lambda(sampling, output_shape=(2,), name='z')([z_mean, z_log_var])
# z is now one n dimensional vector representing the inputs 
encoder_model = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,),)
x = Dense(hidden_dim, activation='relu')(latent_inputs)
outputs = Dense(784, activation='sigmoid')(x)

decoder_model = Model(latent_inputs, outputs, name='decoder')


# instantiate VAE model
outputs = decoder_model(encoder_model(inputs)[2])
vae_model = Model(inputs, outputs, name='vae_mlp')

for i, layer in enumerate(vae_model.layers):
    print("Layer", i+1)
    print("Name", layer.name)
#    print("Input shape", layer.inputs.shape)
#    print("Output shape", layer.outputs.shape)
    if not layer.weights:
        print("No weights for this layer")
        continue
    for i, weight in enumerate(layer.weights):
        print("Weights", i+1)
        print("Name", weight.name)
        print("Weights shape:", weight.shape.as_list())

#Aca hay un error
reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= 784

kl_loss = 0.5 * (K.exp(z_log_var) - (1 + z_log_var) + K.square(z_mean))
kl_loss = K.sum(kl_loss, axis=-1)

total_vae_loss = K.mean(reconstruction_loss + kl_loss)

vae_model.add_loss(total_vae_loss)

vae_model.compile(optimizer='rmsprop',
                  metrics=['accuracy'])
    
vae_model.summary()

vae_model.fit(x_train_flat,
              x_train_flat,
              epochs=epochs,
              batch_size=batch_size)

history = full_model.fit(x_train_flat, x_train_flat, shuffle=True, epochs=1, batch_size=32)

decoded_images = vae_model.predict(x_test_flat)
mse_reconstruction(decoded_images, x_test_flat)

models = encoder_model, decoder_model 
data = x_test_flat, y_test

def plot_results_var(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist",
                    lim=4):
    #Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
    #   models (tuple): encoder and decoder models
    #   data (tuple): test data and label
    #   batch_size (int): prediction batch size
    #   model_name (string): which model is using this function
   

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    
    # display a 2D plot of the digit classes in the latent space
    _, z_log_var, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    print(z_log_var)
    plt.figure(figsize=(8, 7))
    plt.scatter(z_log_var[:, 0], z_log_var[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 10x10 2D manifold of digits
    n = 10
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-1.0 * lim, lim, n)
    grid_y = np.linspace(-1.0 * lim, lim, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(8, 8))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size+1)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

plot_results_var(models,
             data,
             batch_size=batch_size,
             model_name="vae_mlp", 
             lim=5)

loss_ae = train_ae_epochs(10)

vae_mse = []
for i in range(10):
    vae_model.fit(x_train_flat,
            epochs=1,
            batch_size=batch_size)
    decoded_images = vae_model.predict(x_test_flat)
    vae_mse.append(mse_reconstruction(decoded_images, x_test_flat))

plt.plot(range(10), loss_ae/(.01*loss_ae[0]), label='AE')
plt.plot(range(10), vae_mse/(.01*vae_mse[0]), label='VAE')
plt.xlabel('Epochs')
plt.ylabel('MSE Recon. Loss (% of Epoch 1 loss)')
plt.legend()
plt.show()


#######################################################################################################################

                                     #Neuronas Generativas Adversarias (GAN)

#El GAN es una red neuronal
#Son modelos generativos que convierten muestras aleatorias de un distribucion a otra distribucion
#Esto quiere decir que podemos hacer que los datos se adapte o se aproximen a cualquier funcion

#Estos modelos son muy utilizados para crear imagenes ficticios, personajes, etc


import warnings
warnings.simplefilter('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import layers
import time
from tensorflow.keras import models
from tqdm import tqdm

def plot_distribution(real_data,generated_data,discriminator=None,density=True):
    
    plt.hist(real_data.numpy(), 100, density=density, facecolor='g', alpha=0.75, label='real data')
    plt.hist(generated_data.numpy(), 100, density=density, facecolor='r', alpha=0.75,label='generated data q(z) ')
    
    if discriminator:
        max_=np.max([int(real_data.numpy().max()),int(generated_data.numpy().max())])
        min_=np.min([int(real_data.numpy().min()),int(generated_data.numpy().min())])
        x=np.linspace(min_, max_, 1000).reshape(-1,1)
        plt.plot(x,tf.math.sigmoid(discriminator(x,training=False).numpy()),label='discriminator',color='k')
        plt.plot(x,0.5*np.ones(x.shape),label='0.5',color='b')
        plt.xlabel('x')
        
    plt.legend()
    plt.show()

mean = [10]
cov = [[1]]
#Generamos una data con 5000 valores con media=10 y std=1
X = tf.random.normal((5000,1),mean=10,stddev=1.0)

print("mean:",np.mean(X))
print("standard deviation:",np.std(X))

#Generamos otra data con 5000 valores con media=0 y std=2
Z = tf.random.normal((5000,1),mean=0,stddev=2)

print("mean:",np.mean(Z))
print("standard deviation:",np.std(Z))

plot_distribution(X,Z,discriminator=None,density=True)

#Simplemente corremos la data Z 10 para hacer coincidir con la otra
Xhat=Z+10

print("mean:",np.mean(Xhat))
print("standard deviation:",np.std(Xhat))

plot_distribution(X,Xhat,discriminator=None,density=True)


#En GAN hay dos componentes:

#1. El Generador
#2. El Discriminador

#Genera un modelo
def make_generator_model():
    generator = tf.keras.Sequential()
    generator.add(layers.Dense(1))
    return generator

generator=make_generator_model()

#Mas adelante se discute el parametro training=False
Xhat = generator(Z, training=False)

#Grafica las dos datas una real y la generada por Z
plot_distribution(real_data=X, generated_data=Xhat)

###El discriminador

def make_discriminator_model():
    discriminator=tf.keras.Sequential()
    discriminator.add(layers.Dense(1))
    return discriminator

discriminator=make_discriminator_model()

#El generador y el discriminador se inicializan aleatoriamente

#Graficamos las dos datas, y el discriminador (que se ocupa con sigmoid)
plot_distribution(real_data=X,generated_data=Xhat,discriminator=discriminator)

#Muestra los que son mas probables de la funcion real verde que cumplan con el discriminador
py_x=tf.math.sigmoid(discriminator(X,training=False))
print(np.sum(py_x>0.5))

#Muestra los que son mas probables que cumplan con el discriminador
py_x=discriminator(Xhat)
print(np.sum(py_x>0.5))

def get_accuracy(X,Xhat):
    total=0
    py_x=tf.math.sigmoid(discriminator(X,training=False))
    total=np.mean(py_x)
    py_x=tf.math.sigmoid(discriminator(Xhat,training=False))
    total+=np.mean(py_x)
    return total/2

print(get_accuracy(X,Xhat))


###GAN como funcion de perdida

#El gan puede convertir un problema no supervisado a un problema supervisado

#El GAN es bastante dificil de entrenar

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(Xhat):
    return cross_entropy(tf.ones_like(Xhat), Xhat)

def discriminator_loss(X, Xhat):
    real_loss = cross_entropy(tf.ones_like(X), X)
    fake_loss = cross_entropy(tf.zeros_like(Xhat), Xhat)
    total_loss = 0.5*(real_loss + fake_loss)
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(5e-1,beta_1=0.5,beta_2=0.8)

discriminator_optimizer = tf.keras.optimizers.Adam(5e-1,beta_1=0.5, beta_2=0.8)

#paramters for trainng 
epochs=20
BATCH_SIZE=5000
noise_dim=1
epsilon=100 


#discrimator and gernerator 
tf.random.set_seed(0)
discriminator=make_discriminator_model()
generator=make_generator_model()

tf.config.run_functions_eagerly(True)



gen_loss_epoch=[]
disc_loss_epoch=[]
plot_distribution(real_data=X,generated_data=Xhat,discriminator=discriminator )
print("epoch",0)

for epoch in tqdm(range(epochs)):
    #data for the true distribution of your real data samples training ste
    x = tf.random.normal((BATCH_SIZE,1),mean=10,stddev=1.0)
    #random samples it was found if you increase the  stander deviation, you get better results 
    z= tf.random.normal([BATCH_SIZE, noise_dim],mean=0,stddev=10)
    # needed to compute the gradients for a list of variables.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #generated sample 
        xhat = generator(z, training=True)
        #the output of the discriminator for real data 
        real_output = discriminator(x, training=True)
        #the output of the discriminator  data
        fake_output = discriminator(xhat, training=True)
        #loss for each 
        gen_loss= generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    # Compute the gradients for gen_loss and generator
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # Compute the gradients for gen_loss and discriminator
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # Ask the optimizer to apply the processed gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  
  # Save and display the generator and discriminator if the performance increases 
    if abs(0.5-get_accuracy(x,xhat))<epsilon:
        epsilon=abs(0.5-get_accuracy(x,xhat))       
        generator.save('saved_generator.h5')
        discriminator.save('saved_discriminator.h5')
        print(get_accuracy(x,xhat))
        plot_distribution(real_data=X,generated_data=xhat,discriminator=discriminator )
        print("epoch",epoch)
        
generator=make_generator_model()
generator= models.load_model('saved_generator.h5')
xhat=generator(z)
discriminator=models.load_model('saved_discriminator.h5')
plot_distribution(real_data=X,generated_data=xhat,discriminator=discriminator )        


#######################################################################################################################

                                       #Redes Neuronales Convolucionales
                                       #Generativas Adversativas (DGCAN)

#El modelo GAN es muy dificil de entrenar y a menudo las imagenes generadas por GAN son incomprensibles, las redes
#neuronales convolucionales le dan un mejor resultado a GAN

import warnings
warnings.simplefilter('ignore')

import keras

import numpy as np
import tensorflow as tf
print(f"tensorflow version: {tf.__version__}")

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Conv2DTranspose,BatchNormalization,ReLU,Conv2D,LeakyReLU

from IPython import display
import skillsnetwork
print(f"skillsnetwork version: {skillsnetwork.__version__}")

import matplotlib.pyplot as plt
import os
from os import listdir
from pathlib import Path
import imghdr
import os
import time
from tqdm.auto import tqdm

def plot_array(X,title=""):
    
    plt.rcParams['figure.figsize'] = (20,20) 

    for i,x in enumerate(X[0:5]):
        x=x.numpy()
        max_=x.max()
        min_=x.min()
        xnew=np.uint(255*(x-min_)/(max_-min_))
        plt.subplot(1,5,i+1)
        plt.imshow(xnew)
        plt.axis("off")

    plt.show()

img_height, img_width, batch_size=64,64,128

train_ds = tf.keras.utils.image_dataset_from_directory(directory='cartoon_data/cartoon', # change directory to 'cartoon_data' if you use the full dataset
                                                       image_size=(img_height, img_width),
                                                       batch_size=batch_size,
                                                       label_mode=None)

normalization_layer = layers.Rescaling(scale= 1./127.5, offset=-1)
normalized_ds = train_ds.map(lambda x: normalization_layer(x))

images=train_ds.take(1)

X=[x for x in images]

plot_array(X[0])

def make_generator():
    
    model=Sequential()
    
    # input is latent vector of 100 dimensions
    model.add(Input(shape=(1, 1, 100), name='input_layer'))
    
    # Block 1 dimensionality of the output space  64 * 8
    model.add(Conv2DTranspose(64 * 8, kernel_size=4, strides= 4, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1'))
    model.add(ReLU(name='relu_1'))

    # Block 2: input is 4 x 4 x (64 * 8)
    model.add(Conv2DTranspose(64 * 4, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_2'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2'))
    model.add(ReLU(name='relu_2'))

    # Block 3: input is 8 x 8 x (64 * 4)
    model.add(Conv2DTranspose(64 * 2, kernel_size=4,strides=  2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_3'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_3'))
    model.add(ReLU(name='relu_3'))

                       
    # Block 4: input is 16 x 16 x (64 * 2)
    model.add(Conv2DTranspose(64 * 1, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_4'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_4'))
    model.add(ReLU(name='relu_4'))

    model.add(Conv2DTranspose(3, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, 
                              activation='tanh', name='conv_transpose_5'))

    return model

gen = make_generator()
gen.summary()

def make_discriminator():
    
    model=Sequential()
    
    # Block 1: input is 64 x 64 x (3)
    model.add(Input(shape=(64, 64, 3), name='input_layer'))
    model.add(Conv2D(64, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_1'))
    model.add(LeakyReLU(0.2, name='leaky_relu_1'))

    # Block 2: input is 32 x 32 x (64)
    model.add(Conv2D(64 * 2, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_2'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1'))
    model.add(LeakyReLU(0.2, name='leaky_relu_2'))

    # Block 3
    model.add(Conv2D(64 * 4, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_3'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2'))
    model.add(LeakyReLU(0.2, name='leaky_relu_3'))


    #Block 4
    model.add(Conv2D(64 * 8, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_4'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_3'))
    model.add(LeakyReLU(0.2, name='leaky_relu_4'))


    #Block 5
    model.add(Conv2D(1, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False,  
                     activation='sigmoid', name='conv_5'))

    return model 
    
disc = make_discriminator()
disc.summary()    
    
#Definimos funcion de perdida
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(Xhat):
    return cross_entropy(tf.ones_like(Xhat), Xhat)

def discriminator_loss(X, Xhat):
    real_loss = cross_entropy(tf.ones_like(X), X)
    fake_loss = cross_entropy(tf.zeros_like(Xhat), Xhat)
    total_loss = 0.5*(real_loss + fake_loss)
    return total_loss

#Definimos optimizadores

learning_rate = 0.0002

generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999 )

discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999)

@tf.function


#Funcion con los pasos de entrenamiento
def train_step(X):
    
    #random samples it was found if you increase the  stander deviation, you get better results 
    z= tf.random.normal([BATCH_SIZE, 1, 1, latent_dim])
      # needed to compute the gradients for a list of variables.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #generated sample 
        xhat = generator(z, training=True)
        #the output of the discriminator for real data 
        real_output = discriminator(X, training=True)
        #the output of the discriminator for fake data
        fake_output = discriminator(xhat, training=True)
        
        #loss for each 
        gen_loss= generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
      # Compute the gradients for gen_loss and generator
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # Compute the gradients for gen_loss and discriminator
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # Ask the optimizer to apply the processed gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


generator= make_generator()
BATCH_SIZE=128

latent_dim=100
noise = tf.random.normal([BATCH_SIZE, 1, 1, latent_dim])
Xhat=generator(noise,training=False)
plot_array(Xhat)

#Entrenar el modelo para crear imagenes

epochs=1

discriminator=make_discriminator()

generator= make_generator()


for epoch in range(epochs):
    
    #data for the true distribution of your real data samples training ste
    start = time.time()
    i=0
    for X in tqdm(normalized_ds, desc=f"epoch {epoch+1}", total=len(normalized_ds)):
        
        i+=1
        if i%1000:
            print("epoch {}, iteration {}".format(epoch+1, i))
            
        train_step(X)
    

    noise = tf.random.normal([BATCH_SIZE, 1, 1, latent_dim])
    Xhat=generator(noise,training=False)
    X=[x for x in normalized_ds]
    print("orignal images")
    plot_array(X[0])
    print("generated images")
    plot_array(Xhat)
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

#Plotea las imagenes generadas, solo con una epoca no funciono bien, se recomienda ademas ocupar data grandes para que
#pueda hacerlo correctamente

#Se hace un segundo intento con la data completa

###Ejemplo 2

from tensorflow.keras.models import load_model
 
full_generator=load_model("generator")

latent_dim=100

# input consists of noise vectors
noise = tf.random.normal([200, 1, 1, latent_dim])

# feed the noise vectors to the generator
Xhat=full_generator(noise,training=False)
plot_array(Xhat)

for c in [1,0.8,0.6,0.4]:
    Xhat=full_generator(c*tf.ones([1, 1, 1, latent_dim]),training=False) # latent_dim = 100 defined previously
    plot_array(Xhat)

for c in [1,0.8,0.6,0.4]:
    Xhat=full_generator(-c*tf.ones([1, 1, 1, latent_dim]),training=False)
    plot_array(Xhat)

z=np.ones( (1, 1, 1, latent_dim))
for n in range(10):

    z[0, 0, 0, 0:10*n]=-1

    Xhat=full_generator(z,training=False)
    print("elements from 0 to {} is set to -1".format(10*n))
    plot_array(Xhat)

z=np.ones( (1, 1, 1, latent_dim))
for n in range(5):

    z[0, 0, 0, 0:20*n]=-0.5*n

    Xhat=full_generator(z,training=False)

    plot_array(Xhat)

for n in range(10):
    z=np.random.normal(0, 1, (1, 1, 1, latent_dim))

    z[0,0,0,0:35]=1

    Xhat=full_generator(z,training=False)

    plot_array(Xhat)

for n in range(10):
    z=np.random.normal(0, 1, (1, 1, 1, latent_dim))

    z[0,0,0,0:35]=-1

    Xhat=full_generator(z,training=False)

    plot_array(Xhat)


#######################################################################################################################

                                         #Autocodificadores Variacionales
                                           #Hay errores en el codigo

#Los autocodificadores variacionales son para generar musica, digitos, imagenes y lo que se te ocurra

#La arquitectura es similar a la de un autocodificador tradicional con un codificador y decodificador, ambos optimizados
#con el menor error posible en la reconstruccion

#Para poder usar VAE para generar, debemos usarlo con regulacion en el espacio latente

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
import numpy as np

# Import the keras library
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Layer,Reshape,Conv2DTranspose
from tensorflow.python.client import device_lib
from keras.layers import Multiply, Add
from keras import backend as K

from numpy import random
import numpy as np
from matplotlib import pyplot as plt

def plot_label_clusters(model, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ =encoder.predict(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

(X_train, y_train), (_, _) = keras.datasets.mnist.load_data()

print(X_train.shape, y_train.shape)

print(np.unique(y_train))

print(f"Before reshaping, X_train has a shape of: {X_train.shape}")

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
print(f"After reshaping, X_train has a shape of: {X_train.shape}")

X_train = tf.cast(X_train, tf.float32)
X_train = X_train/255.0

dataset=tf.data.Dataset.from_tensor_slices(X_train)
print(dataset)

for r in random.randint(0, 59999, size=5, dtype=int):
    
    plt.figure(figsize=(3,3))
    plt.imshow(X_train[r,:,:,0],cmap="gray")
    plt.title("sample No.{} belongs to class {}".format(r,y_train[r]))
    plt.axis("off")
plt.show()    

#Creamos un codificador con dos capas ocultas
encoder_input= keras.Input(shape=(28, 28, 1))
x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_input)
x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = Flatten()(x)
encoder_output = Dense(16, activation="relu")(x)    
    
#El output de la segunda capa, hace que los datos salgan con una cierta distribucion normal, una media y una std

#El siguiente paso aprende de esa media y esa std

latent_dim = 2

# Dense layer to learn the mean
mean = Dense(latent_dim, name="mean")(encoder_output)
# Dense layer to learn the log variance
log_var = Dense(latent_dim, name="z_log_var")(encoder_output)
class MyLayer(Layer):
    def call(self, x):
        return tf.exp(0.5*x)

# sigmia is calculated from log variance
sigma=MyLayer()(log_var)
class MyLayer2(Layer):
    def call(self, x):
        return tf.shape(x)
epsilon_shape = MyLayer2()(mean)
epsilon = tf.keras.backend.random_normal(shape = epsilon_shape)
#epsilon = tf.keras.backend.random_normal(shape = (tf.shape(mean)[0], tf.shape(mean)[1]))
#epsilon = tf.keras.backend.random_normal(shape = (mean.shape[0], mean.shape[1]))
#z = mean + sigma * epsilon 
z_eps = Multiply()([sigma, epsilon])
z = Add()([mean, z_eps])

encoder = Model(encoder_input, outputs = [mean, log_var, z], name = 'encoder')
encoder.summary()

#Ahora la parte de decodificador

latent_dim=2
decoder=Sequential()

decoder.add(keras.Input(shape=(latent_dim,))) # input dimension is 2
decoder.add(Dense(7 * 7 * 64, activation="relu"))
decoder.add(Reshape((7, 7, 64)))
decoder.add(Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))
decoder.add(Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))
decoder.add(Conv2DTranspose(1, 3, activation="sigmoid", padding="same"))
decoder.summary()


#Incluimos la funcion de perdida que mide la reconstruccion
#utilizando el error cuadratico medio

#reconstruction_loss
def reconstruction_loss(y, y_hat):
    return tf.reduce_mean(tf.square(y - y_hat))


#Kullback–Leibler divergence encoder loss
def kl_loss(mu, log_var):
    loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    return loss

# add two losses 
def vae_loss(y_true, y_hat, mu, log_var):
    return reconstruction_loss(y_true, y_hat) + (1 / (64*64)) * kl_loss(mean, log_var)

# encoder returns mean and log variance of the normal distribution,
# and a sample point z
mean, log_var, z = encoder(encoder_input)

# decoder decodes the sample z 
reconstructed = decoder(z)

model = Model(encoder_input, reconstructed, name ="vae")
loss = kl_loss(mean, log_var)
model.add_loss(loss)
model.summary()



#Entrenamos la VAE

#loss
mse_losses = []
kl_losses = []
# optimizer 
optimizer =  tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999 )
epochs = 5

for epoch in range(epochs):
    
    print(f"Samples generated by decoder after {epoch} epoch(s): ")
    
    #random noise
    z = tf.random.normal(shape = (5, latent_dim,))

    # input random noise into the decoder
    xhat = decoder.predict(z)
    
    # plot the decoder output
    plt.figure()
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(xhat[i,:,:,0],cmap="gray")
        plt.axis("off")
    plt.show()

    print(f"2D latent representations of the training data produced by encoder after {epoch} epoch(s): ")
    plot_label_clusters(encoder, X_train, y_train)


    # training steps
    for (step, training_batch) in enumerate(dataset.batch(100)):
        with tf.GradientTape() as tape:

            # model output
            reconstructed = model(training_batch)

            y_true = tf.reshape(training_batch, shape = [-1])
            y_pred = tf.reshape(reconstructed, shape = [-1])

            # calculate reconstruction loss
            mse_loss = reconstruction_loss(y_true, y_pred)
            # calculate KL divergence
            kl = sum(model.losses)

            kl_losses.append(kl.numpy())
            mse_losses.append(mse_loss .numpy())

            # total loss
            train_loss = 0.01 * kl + mse_loss

            grads = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
         
    print("Epoch: %s - Step: %s - MSE loss: %s - KL loss: %s" % (epoch, step, mse_loss.numpy(), kl.numpy()))

plt.plot(mse_losses)
plt.title(" reconstruction loss ")
plt.show()
plt.plot(kl_losses)
plt.title("  Kullback–Leibler divergence")
plt.show()

xhat = decoder.predict(z)

for i in range(5):
    plt.imshow(xhat[i,:,:,0],cmap="gray")
    plt.show()
    
plot_label_clusters(encoder, X_train, y_train)  

 
#######################################################################################################################
  
                                            #GPU y CPU en Keras 
                 #GPU (Unidad de procesamiento graficos), CPU (Unidad central de procesamiento)

#La CPU es el procesador del pc, independiente de la tecnologia del procesador

#Los GPU son especialmente utiles para acelerar ciertos procesos, se puede aprovechar el uso en paralelo de los
#calculos de entrenamiento, las GPU se ocupara para hacer procesamientos GRAFICOS es decir tienen mucha eficiencia
#con las imagenes y videos, su principal potencia se ve en REDES CONVOLUCIONALES

#GPU es muy eficientemente con el uso de redes neuronales en comparacion con CPU

import warnings
warnings.simplefilter('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np

import tensorflow as tf
# Import the keras library
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.client import device_lib

###Como hacer que ignore todos los GPU y solo tome los CPU

#-1 significa que se ejecutara con CPU
#Esto se debe ejecutar antes de importar tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#0,1,2 o cualquier positivo indica que ocupara GPU
#Esto se debe ejecutar antes de importar tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

###Ejemplo de como utilizar CPU

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

y_train = y_train.reshape((y_train.shape[0],1))
y_test = y_test.reshape((y_test.shape[0],1))

#Aca construimos el modelo especificando que queremos que ocupe el dispositivo CPU
with tf.device('/CPU:0'):
    model_cpu = Sequential()
    model_cpu.add(Conv2D(input_shape = (28, 28, 1),
                     filters=5, 
                     padding='Same',
                     kernel_size=(3,3)
                     ))
    model_cpu.add(MaxPooling2D(pool_size=(2,2)))
    model_cpu.add(Flatten())
    model_cpu.add(Dense(256, activation='relu'))
    model_cpu.add(Dense(10, activation='softmax'))
    
    model_cpu.compile(optimizer='adam', 
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model_cpu.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)


#Aca construimos un modelo especificando que queremos que ocupe el dispositivo GPU

#Primero verificamos si tenemos GPU
#Si ocupamos un notebook la tarjeta grafica de base no tendremos GPU, pero en pc armado si podemos ponerle una GPU
#es un dispositivo, pero es bastante grande y necesitan su ventilacion propia

print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Si es que tenemos GPU podemos ejecutarlas en un orden en particular
print(device_lib.list_local_devices())

with tf.device('/device:GPU:2'):
    model_gpu = Sequential()
    model_gpu.add(Conv2D(input_shape = (28, 28, 1),
                     filters=5, 
                     padding='Same',
                     kernel_size=(3,3)
                     ))
    model_gpu.add(MaxPooling2D(pool_size=(2,2)))
    model_gpu.add(Flatten())
    model_gpu.add(Dense(256, activation='relu'))
    model_gpu.add(Dense(10, activation='softmax'))
    
    model_gpu.compile(optimizer='adam', 
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model_gpu.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)


#Ejemplo de mezclar CPU y GPU

# Enable tensor allocations or operations to be printed
tf.debugging.set_log_device_placement(True)

# Get list of all logical GPUs
gpus = tf.config.list_logical_devices('GPU')

# Check if there are GPUs on this computer
if gpus:
  # Run matrix computation on multiple GPUs
    c = []
    for gpu in gpus:
        with tf.device(gpu.name):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) 
            c.append(tf.matmul(a, b))

    # Run on CPU 
    with tf.device('/CPU:0'):
        matmul_sum = tf.add_n(c)

    print(matmul_sum)


#######################################################################################################################

                                           #Aprendizaje por reforzamiento
                                             #Existen algunos errores

#El aprendizaje de reforzamiento se refiere al aprendizaje que es por acciones, es decir por ejemplo en un juego si
#el jugador hace algo, la inteligencia responde automaticamente esa accion (recompensa) y le devuelve otra accion al jugador
#La maquina va aprendiendo de estas acciones


import gym
import pandas
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def warn(*args, **kwargs):
    return None

warnings.warn = warn

# Contruye un envolvente para los calculos y algoritmos, etc gym.make()
env = gym.make('FrozenLake-v1') # Build a fresh environment

#Empieza de nuevo el envolvente env.reset()
current_observation = env.reset() # This starts a new "episode" and returns the initial observation

# The current observation is just the current location
print(current_observation) # Observations are just a number

# we can print the environment if we want to look at it
env.render()

# the action space for this environment includes four discrete actions

print(f"our action space: {env.action_space}")

new_action = env.action_space.sample() # we can randomly sample actions

print(f"our new action: {new_action}") # run this cell a few times to get an idea of the action space
# what does it look like?

# now we act! do this with the step function

new_action = env.action_space.sample()

observation, reward, terminated, truncated, info = env.step(new_action)
done= terminated or truncated
# here's a look at what we get back
print(f"observation: {observation}, reward: {reward}, done: {done}, info: {info}")

env.render() 

# we can put this process into a for-loop and see how the game progresses

current_observation = env.reset() # start a new game

for i in range(5): # run 5 moves

    new_action = env.action_space.sample() # same a new action

    observation, reward, terminated, truncated, info = env.step(new_action) # step through the action and get the outputs
    done=terminated or truncated
    # here's a look at what we get back
    print(f"observation: {observation}, reward: {reward}, done: {done}, info: {info}")

    env.render() 

#El entorno define tambien acciones y diferentes resultados

#Definimos que son algunos parametros
#1.Observation= Se refiere al numero de baldosas 0,1,2,3,
#                                                4,5,6...
#2.Reward= Se refiere al resultado del juego
#3.Done=Nos dice si el juego aun continuo a termina
#4.Info=Indica informacion adicional sobre el mundo con probabilidades 

# Here's how to simulate an entire episode
# We're going to stop rendering it every time to save space
# try running this a few. Does it ever win?

current_observation = env.reset()
done = False

while not done:    
    new_action = env.action_space.sample()
    new_observation, reward, terminated, truncated, info = env.step(new_action)
    done=terminated or truncated
    print(f"action:{new_action} observation: {new_observation}, reward: {reward}, done: {done}, info: {info}")

#Definiciones de funciones
#initial_observation = env.reset() #Se ocupa para empezar un nuevo episodio y retorna la observacion inicial
#new_observation, reward, done, info = env.step(new_action)  #Ejecuta una nueva accion y devuelve una nueva observavion
#done != True  #Hasta que finalize el juego

#Se intentan maximizar las recompensas por lo que la primera recompenza empieza en 0
#env.action_space.n  #Proporciona el numero de acciones posibles en el entorno
#env.action_space.sample()  #Permite muestrear aleatoriamente una accion
#env.observation_space.n  #Proporciona el numero de estados posibles en el entorno

env = gym.make('FrozenLake-v1')

num_episodes = 40000

life_memory = []
for i in range(num_episodes):
    
    # start a new episode and record all the memories
    old_observation = env.reset()
    done = False
    tot_reward = 0
    ep_memory = []
    while not done:
        new_action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(new_action)
        done=terminated or truncated
        tot_reward += reward
        
        ep_memory.append({
            "observation": old_observation,
            "action": new_action,
            "reward": reward,
            "episode": i,
        })
        old_observation = observation
        
    # incorporate total reward
    num_steps = len(ep_memory)
    for i, ep_mem in enumerate(ep_memory):
        ep_mem["tot_reward"] = tot_reward
        ep_mem["decay_reward"] = i*tot_reward/num_steps
        
    life_memory.extend(ep_memory)
    
memory_df = pandas.DataFrame(life_memory)

print(memory_df.describe())

print(memory_df.shape)

print(memory_df.groupby("episode").reward.sum().mean())




#observations = np.array([flatten_observation(obs) for obs in memory_df["observation"]])
#actions = np.array(memory_df["action"]).reshape(-1, 1)

###Prediccion

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

model = ExtraTreesRegressor(n_estimators=50)
# model = SVR()
y = 0.5*memory_df.reward + 0.1*memory_df.decay_reward + memory_df.tot_reward
x = memory_df[["observation", "action"]]
model.fit(x, y)

#Ahora existe un modelo que predice el comportamiento deseado

#Empezamos ganando el 1.5% de los juegos

model = RandomForestRegressor()
y = 1*memory_df.reward + memory_df.tot_reward + .1*memory_df.decay_reward
x = memory_df[["observation", "action"]]
model.fit(x, y)

num_episodes = 500
random_per = 0

life_memory = []
for i in range(num_episodes):
    # Start a new episode and record all the memories.
    old_observation = env.reset()
    done = False
    tot_reward = 0
    ep_memory = []
    while not done:
        if np.random.rand() < random_per:
            new_action = env.action_space.sample()
        else:
            pred_in = [[old_observation,i] for i in range(4)]
            new_action = np.argmax(model.predict(pred_in))
        observation, reward, done, info = env.step(new_action)
        tot_reward += reward
        
        ep_memory.append({
            "observation": old_observation,
            "action": new_action,
            "reward": reward,
            "episode": i,
        })
        old_observation = observation
        
    # incorporate total reward
    for ep_mem in ep_memory:
        ep_mem["tot_reward"] = tot_reward
        
    life_memory.extend(ep_memory)
    
memory_df2 = pandas.DataFrame(life_memory)

# rf.fit(memory_df[["observation", "action"]], memory_df["comb_reward"])

# Score
# Much better!
print(memory_df2.groupby("episode").reward.sum().mean())

y = .1*memory_df.reward + 1*memory_df.decay_reward + 1*memory_df.tot_reward


###Extension carro de postes

env = gym.make('CartPole-v1')

# now we can build a toy world!
num_episodes = 1000

life_memory = []
for i in range(num_episodes):
    
    # start a new episode and record all the memories
    old_observation = env.reset()
    done = False
    tot_reward = 0
    ep_memory = []
    while not done:
        new_action = env.action_space.sample()
        observation, reward, done, info = env.step(new_action)
        tot_reward += reward
        
        ep_memory.append({
            "obs0": old_observation[0],
            "obs1": old_observation[1],
            "obs2": old_observation[2],
            "obs3": old_observation[3],
            "action": new_action,
            "reward": reward,
            "episode": i,
        })
        old_observation = observation
        
    # incorporate total reward
    for ep_mem in ep_memory:
        ep_mem["tot_reward"] = tot_reward
        
    life_memory.extend(ep_memory)
    
memory_df = pandas.DataFrame(life_memory)

memory_df.groupby("episode").reward.sum().mean()

print(memory_df.describe())

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor

model = ExtraTreesRegressor(n_estimators=50)

memory_df["comb_reward"] = .5*memory_df.reward + memory_df.tot_reward
model.fit(memory_df[["obs0", "obs1", "obs2", "obs3", "action"]], memory_df.comb_reward)

num_episodes = 100
random_per = 0

life_memory = []
for i in range(num_episodes):
    
    # start a new episode and record all the memories
    old_observation = env.reset()
    done = False
    tot_reward = 0
    ep_memory = []
    while not done:
        
        
        if np.random.rand() < random_per:
            new_action = env.action_space.sample()
        else:
            pred_in = [list(old_observation)+[i] for i in range(2)]
            new_action = np.argmax(model.predict(pred_in))
        observation, reward, done, info = env.step(new_action)
        tot_reward += reward
        
        ep_memory.append({
            "obs0": old_observation[0],
            "obs1": old_observation[1],
            "obs2": old_observation[2],
            "obs3": old_observation[3],
            "action": new_action,
            "reward": reward,
            "episode": i,
        })
        old_observation = observation
        
    # incorporate total reward
    for ep_mem in ep_memory:
        ep_mem["tot_reward"] = tot_reward
        
    life_memory.extend(ep_memory)
    
memory_df2 = pandas.DataFrame(life_memory)
memory_df2["comb_reward"] = memory_df2.reward + memory_df2.tot_reward

# score
# much better!
print(memory_df2.groupby("episode").reward.sum().mean())





