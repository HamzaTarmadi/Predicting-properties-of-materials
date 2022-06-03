'''
---------- A Lire ----------

Ce fichier permet de récuperer X et Y produits par le fichier B2, de 
partitionner les données en un ensemble de test et un d'entraînement puis
d'entraîner un modèle dessus et de voir ses performances sur l'ensemble de
test.

En particulier, on a essayé d'utiliser keras et MLPRegressor de sklearn.

Malheureusement, cela ne fonctionne pas, on obtient pas de résultats : toutes
les valeurs prédites sont les mêmes.
Je ne comprends pas ce qui ne fonctionne pas.
'''

import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pymatgen.io.cif import CifParser

from dscribe.descriptors import SOAP
#from ase.build import bulk
from ase import Atoms
from ase.visualize import view
import scipy

#==========# Récapitulatif des paramètres lors execution #==========#

#ProportionBaseDonnee :
#ProportionTrainTest :
#alpha :
#activation function :
#hidden_layer_sizes :
#solver : 

#==========# Récupération données #==========#

def randomisation(x, y):
    y = y.reshape(-1,1)
    total = np.concatenate([x, y], axis = 1)
    np.random.shuffle(total)
    x = total[:,:-1]
    y = total[:, -1]
    return x, y.reshape(-1)

proportionBaseDonnee = 0.10
proportionTrainTest = 0.75

print('Downloading x and y...')

y = np.loadtxt('baseDonneeY2.csv', delimiter=',')
x = np.loadtxt('baseDonneeX2.csv', delimiter=',')
# x a shape (len(baseDonnee), nbFeatures) C'est une COO sparse Matrix
#x = x.tocsr()

x, y = randomisation(x, y)

print('x and y downloaded')
print(f'x.shape : {x.shape}')
print(f'y.shape : {y.shape}')
print(f'proportionBaseDonnee : {proportionBaseDonnee}')
print(f'proportionTrainTest : {proportionTrainTest}')

nbInstances = int(proportionBaseDonnee*y.shape[0])
y = y[:nbInstances]
x = x[:nbInstances,:]

idPremierElemTEST = int(nbInstances * proportionTrainTest)

xTRAIN = x[:idPremierElemTEST,:]
xTEST = x[idPremierElemTEST:,:]

yTRAIN = y[:idPremierElemTEST]
yTEST = y[idPremierElemTEST:]

print('Train and test datasets separated')

#==========# Regression - fittage #==========#

print('Creating regressor...')

#Premier modèle
'''
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning

clf = MLPRegressor(solver='adam', alpha=1e-5, 
                   hidden_layer_sizes=(1000, 100, 100, 1000), 
                   random_state=0, max_iter=100,
                   verbose=True, early_stopping=False,
                   activation = 'logistic')
print('Regressor created')

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    clf.fit(xTRAIN, yTRAIN)
print('Model fitted')

yPredictTEST = clf.predict(xTEST)
'''

#Deuxième modèle

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
  
model = keras.Sequential([
      layers.Input(xTRAIN.shape[1],),
      layers.Dense(1024, activation='relu'),
      layers.Dense(1024, activation='relu'),
      layers.Dense(256, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(1)
  ])

model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
print(model.summary())

history = model.fit(xTRAIN, yTRAIN,  verbose=1, epochs=20)

yPredictTEST = model.predict(xTEST)


#==========# Prédiction - Affichage #==========#

def printTabTup(tab1, tab2):
    for i in range(len(tab1)):
        print(f'{i} - real value : {tab1[i]} - predicted value : {tab2[i]}')

xId = [i for i in range(-4, 4)]
yId = [i for i in range(-4, 4)]

r2 = 0 #clf.score(xTEST, yTEST)

fig, axs = plt.subplots()
axs.plot(xId, yId, 'red')
axs.scatter(yPredictTEST, yTEST, s = 3)

axs.set(ylabel = 'enerForMoyAtom prédite', xlabel = 'enerForMoyAtom réelle')
fig.suptitle(f'Energie formation moyenne prédite / réelle \n R2 = {r2}')

plt.show()

