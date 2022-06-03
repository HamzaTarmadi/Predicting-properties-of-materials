'''
---------- A Lire ----------

Ce fichier python peut être executé directement : associé au fichier 
dataFormEnCif_with_Carbone.txt permet la création d'un bon modèle de 
prédiction.
Affiche la courbe des energies de formation moyennes en fonction des réelles 
sur des données de test.

Il est brut et pas organisé sous forme de fonctions claires. L'objectif 
était d'avoir un modèle rapidement pour voir si poursuivre dans cette voie
avec le SOAP était pertinent.

Sur un PC moyen, prend en l'état de l'ordre de quelques minutes à quelques 
dizaines de minutes.
Possibilité de changer nmax, lmax, et la structure du réseau de neurones.
'''

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pymatgen.io.cif import CifParser

from dscribe.descriptors import SOAP
#from ase.build import bulk
from ase import Atoms
from ase.visualize import view

#==========# Chargement des données #==========#

f = open("dataFormEnCif_with_Carbone.txt", "r")
truc = json.loads(f.read())
truc2 = pd.DataFrame(truc['response'])

#==========# Constantes #==========#

species = ['H','Li','B','C','N','O','F','Na','Mg','Al','Si',
            'P','S','Cl','K','Ca','Ti','V','Cr','Mn','Fe','Co',
            'Ni','Cu','Zn','Se','Sr','Ba']
rcut = 6.0
nmax = 4
lmax = 3

dictionnaireTabPeriodique = {'H' : 1, 'He' : 2, 'Li' : 3, 'Be' : 4, 
                            'B' : 5, 'C' : 6, 'N' : 7, 'O' : 8, 
                            'F' : 9, 'Ne' : 10, 'Na' : 11, 'Mg' : 12, 
                            'Al' : 13, 'Si' : 14, 'P' : 15, 'S' : 16, 
                            'Cl' : 17, 'Ar' : 18, 'K' : 19, 'Ca' : 20, 
                            'Sc' : 21, 'Ti' : 22, 'V' : 23, 'Cr' : 24, 
                            'Mn' : 25, 'Fe' : 26, 'Co' : 27, 'Ni' : 28, 
                            'Cu' : 29, 'Zn' : 30, 'Ga' : 31, 'Ge' : 32, 
                            'As' : 33, 'Se' : 34, 'Br' : 35, 'Kr' : 36, 
                            'Rb' : 37, 'Sr' : 38, 'Y' : 39, 'Zr' : 40, 
                            'Nb' : 41, 'Mo' : 42, 'Tc' : 43, 'Ru' : 44, 
                            'Rh' : 45, 'Pd' : 46, 'Ag' : 47, 'Cd' : 48, 
                            'In' : 49, 'Sn' : 50, 'Sb' : 51, 'Te' : 52, 
                            'I' : 53, 'Xe' : 54, 'Cs' : 55, 'Ba' : 56, 
                            'La' : 57, 'Ce' : 58, 'Pr' : 59, 'Nd' : 60, 
                            'Pm' : 61, 'Sm' : 62, 'Eu' : 63, 'Gd' : 64, 
                            'Tb' : 65, 'Dy' : 66, 'Ho' : 67, 'Er' : 68, 
                            'Tm' : 69, 'Yb' : 70, 'Lu' : 71, 'Hf' : 72, 
                            'Ta' : 73, 'W' : 74, 'Re' : 75, 'Os' : 76, 
                            'Ir' : 77, 'Pt' : 78, 'Au' : 79, 'Hg' : 80, 
                            'Tl' : 81, 'Pb' : 82, 'Bi' : 83, 'Po' : 84, 
                            'At' : 85, 'Rn' : 86, 'Fr' : 87, 'Ra' : 88, 
                            'Ac' : 89, 'Th' : 90, 'Pa' : 91, 'U' : 92, 
                            'Np' : 93, 'Pu' : 94, 'Am' : 95, 'Cm' : 96, 
                            'Bk' : 97, 'Cf' : 98, 'Es' : 99, 'Fm' : 100, 
                            'Md' : 101, 'No' : 102, 'Lr' : 103, 'Rf' : 104, 
                            'Db' : 105, 'Sg' : 106, 'Bh' : 107, 'Hs' : 108, 
                            'Mt' : 109, 'Ds' : 110, 'Rg' : 111, 'Cn' : 112, 
                            'Nh' : 113, 'Fl' : 114, 'Mc' : 115, 'Lv' : 116, 
                            'Ts' : 117, 'Og' : 118}

#==========# Création classe SOAP #==========#

periodic_soap = SOAP(
    species=species,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    periodic=True,
    sparse=False
)

#==========# Calcul des SOAP #==========#

tabSoap = []
compteur = 0
tabForEn = []

print('Creating Soaps...')

for i in range(2000):
    if i%100 == 0:print(f'{i} / 2000')
    blurp = CifParser.from_string(truc['response'][i]['cif'])
    blurpDict = blurp.as_dict()
    blurpDict = blurpDict.popitem()[1]
    #Constantes
    a = float(blurpDict['_cell_length_a'])
    b = float(blurpDict['_cell_length_b'])
    c = float(blurpDict['_cell_length_c'])
    alpha = float(blurpDict['_cell_angle_alpha'])
    beta = float(blurpDict['_cell_angle_beta'])
    gamma = float(blurpDict['_cell_angle_gamma'])
    #Cell
    in_cell = [a, b, c, alpha, beta, gamma]
    #Numbers
    try:
        in_numbers = [dictionnaireTabPeriodique[i] 
                      for i in blurpDict['_atom_site_type_symbol']]
    except:
        continue
    #Positions
    tabX = np.array(blurpDict['_atom_site_fract_x']).astype(float)
    tabY = np.array(blurpDict['_atom_site_fract_y']).astype(float)
    tabZ = np.array(blurpDict['_atom_site_fract_z']).astype(float)
    tabX = tabX.reshape((1,tabX.shape[0]))
    tabY = tabY.reshape((1,tabY.shape[0]))
    tabZ = tabZ.reshape((1,tabZ.shape[0]))
    in_positions = np.concatenate(
                (tabX, 
                 tabY,
                 tabZ
                 ),
                axis = 0).transpose()
    try:
        maille = Atoms(cell = in_cell,
                       numbers = in_numbers,
                       positions = np.zeros(in_positions.shape),
                       pbc=[True, True, True])
        maille.set_scaled_positions(in_positions)
        soap_maille = periodic_soap.create(maille)
        nbAtomes = len(blurpDict['_atom_site_fract_x'])
        soap_maille = soap_maille.sum(axis=0).reshape(1, -1) / nbAtomes
        tabSoap.append(soap_maille)
        tabForEn.append(truc['response'][i]['formation_energy_per_atom'])
    except ValueError:
        compteur +=1
        #print('compteur : ', compteur)

#==========# Regression - fittage #==========#

from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning

yTrain = tabForEn
xTrain = np.concatenate(tabSoap)

print('Creating regressor')
clf = MLPRegressor(solver='adam', alpha=1e-5, 
                   hidden_layer_sizes=(1000, 100, 50, 100, 100, 1000), 
                   random_state=1, max_iter=100,
                   verbose=True, early_stopping=False,
                   activation = 'logistic')
print('regressor created')

print('fitting model')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    clf.fit(xTrain, yTrain)
print('model fitted')

#fig, axs = plt.subplots()
#axs.scatter(x, y, s = 4)
#axs.scatter(range(400), clf.predict([[i] for i in range(400)]), s = 4)

#plt.show()

#==========# Test du modèle #==========#

compteur2 = 0
tabForEnTest = []
tabSoapTest = []

for i in range(2001, 3000):
    #print(i)
    blurp = CifParser.from_string(truc['response'][i]['cif'])
    blurpDict = blurp.as_dict()
    blurpDict = blurpDict.popitem()[1]
    #Constantes
    a = float(blurpDict['_cell_length_a'])
    b = float(blurpDict['_cell_length_b'])
    c = float(blurpDict['_cell_length_c'])
    alpha = float(blurpDict['_cell_angle_alpha'])
    beta = float(blurpDict['_cell_angle_beta'])
    gamma = float(blurpDict['_cell_angle_gamma'])
    #Cell
    in_cell = [a, b, c, alpha, beta, gamma]
    #Numbers
    try:
        in_numbers = [dictionnaireTabPeriodique[i] 
                      for i in blurpDict['_atom_site_type_symbol']]
    except:
        continue
    #Positions
    tabX = np.array(blurpDict['_atom_site_fract_x']).astype(float)
    tabY = np.array(blurpDict['_atom_site_fract_y']).astype(float)
    tabZ = np.array(blurpDict['_atom_site_fract_z']).astype(float)
    tabX = tabX.reshape((1,tabX.shape[0]))
    tabY = tabY.reshape((1,tabY.shape[0]))
    tabZ = tabZ.reshape((1,tabZ.shape[0]))
    in_positions = np.concatenate(
                (tabX, 
                 tabY,
                 tabZ
                 ),
                axis = 0).transpose()
    try:
        maille = Atoms(cell = in_cell,
                       numbers = in_numbers,
                       positions = np.zeros(in_positions.shape),
                       pbc=[True, True, True])
        maille.set_scaled_positions(in_positions)
        soap_maille = periodic_soap.create(maille)
        nbAtomes = len(blurpDict['_atom_site_fract_x'])
        soap_maille = soap_maille.sum(axis=0).reshape(1, -1) / nbAtomes
        formEnPred = clf.predict(soap_maille)
        tabForEnTest.append((truc['response'][i]['formation_energy_per_atom'], formEnPred))
        tabSoapTest.append(soap_maille)
    except ValueError as er:
        #print(er)
        compteur2 += 1
        #print('compteur : ', compteur2)

#==========# Affichage #==========#

def printTabTup(tab):
    for i in range(len(tab)):
        print(f'{i} - real value : {tab[i][0]} - predicted value : {tab[i][1]}')

printTabTup(tabForEnTest)   

trueEnergie = [tabForEnTest[i][0] for i in range(len(tabForEnTest))]
predictEnergie = [tabForEnTest[i][1] for i in range(len(tabForEnTest))]

xTest = np.concatenate(tabSoapTest)

x2 = [i for i in range(-4, 4)]
y2 = [i for i in range(-4, 4)]

r2 = clf.score(xTest, trueEnergie)

fig, axs = plt.subplots()
axs.plot(x2, y2, 'red')
axs.scatter(trueEnergie, predictEnergie, s = 4)
axs.set(ylabel = 'enerForMoyAtom prédite', xlabel = 'enerForMoyAtom réelle')
fig.suptitle('R2 = {r2}')

plt.show()

#==========# Sauvegarde bouts de code #==========#

#print(soap_maille.sum(axis=1))

'''test = Atoms( 
               cell=[1.63153115, 1.63153115, 9.87987700, 
                     90.00000000, 90.00000000, 90.19895434],
               numbers=[6, 6],
               positions=[[0.33121500*1.63153115, 0.33121500*1.63153115, 0.75000000*9.87987700],
                          [0.66878500*1.63153115, 0.66878500*1.63153115, 0.25000000*9.87987700]],
               pbc=True)'''


#Matrice de changement de bases
"""
def f(a, b, c, alpha, beta, gamma):
    numerateur = 1 - np.cos(beta)**2 - np.cos(alpha)**2
    produit_triple = np.cos(beta)*np.cos(alpha)*np.cos(gamma)
    denominateur = np.sqrt(1 - np.cos(beta)**2 - np.cos(alpha)**2 + 2*produit_triple)
    return numerateur / denominateur

Mat = np.zeros((3,3))
Mat[0, 0] = 1
Mat[0, 1] = np.cos(gamma)
Mat[1, 1] = np.sin(gamma)
Mat[0, 2] = np.cos(beta)
Mat[1, 2] = (np.cos(alpha) / np.sin(gamma) - np.cos(beta) / np.tan(gamma))
Mat[2, 2] = f(a, b, c, alpha, beta, gamma)

Mat[:, 0] *= a
Mat[:, 1] *= b
Mat[:, 2] *= c

MatInv = np.linalg.inv(Mat)
"""