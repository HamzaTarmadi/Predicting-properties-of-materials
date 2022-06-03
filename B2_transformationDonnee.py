'''
---------- A Lire ----------

Ce fichier permet de transformer les données cif regroupées dans un format
json en tableaux X et Y, respectivement matrice des observations avec la
représentation SOAP et vecteur des energie de formation correspondantes.

En particulier, on sauvegarde X sous forme de sparse.matrix.

Les transformations sont permises par les fonctions :
    extractCifDict
    computeMaille
    computeSoapReduced
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

#==========# Chargement des données #==========#

f = open("dataFormEnCifGrand.txt", "r")
baseDonnee = json.loads(f.read())

#==========# Constantes #==========#

species = ['H','Li','B','C','N','O','F','Na','Mg','Al','Si',
            'P','S','Cl','K','Ca','Ti','V','Cr','Mn','Fe','Co',
            'Ni','Cu','Zn','Se','Sr','Ba']
rcut = 6.0
nmax = 3
lmax = 2

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

print('creation of the SOAP class ...')

periodic_soap = SOAP(
    species=species,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    periodic=True,
    sparse=False #format COO
)

print('SOAP created')

#==========# Calcul des SOAP #==========#

def extractCifDict(i):
    '''
    Parameters
    ----------
    i : int
        Indice entre 0 et len(baseDonnee['response']).

    Returns
    -------
    instanceCifDict : dict
        Contient le cif de l'instance i sous forme de dictionnaire.
        
    '''
    instanceCifPars = CifParser.from_string(baseDonnee['response'][i]['cif'])
    instanceCifDict = instanceCifPars.as_dict()
    instanceCifDict = instanceCifDict.popitem()[1]
    return instanceCifDict

def computeMaille(instanceCifDict):
    '''
    Parameters
    ----------
    instanceCifDict : dict
        Contient le cif de l'instance i sous forme de dictionnaire.

    Returns
    -------
    maille : ase.Atoms
        La maille associée au cif au format ase.Atoms.
        
    '''
    #Constantes
    a = float(instanceCifDict['_cell_length_a'])
    b = float(instanceCifDict['_cell_length_b'])
    c = float(instanceCifDict['_cell_length_c'])
    alpha = float(instanceCifDict['_cell_angle_alpha'])
    beta = float(instanceCifDict['_cell_angle_beta'])
    gamma = float(instanceCifDict['_cell_angle_gamma'])
    #Cell
    in_cell = [a, b, c, alpha, beta, gamma]
    #Numbers
    in_numbers = [dictionnaireTabPeriodique[i] 
                      for i in instanceCifDict['_atom_site_type_symbol']]
    #Positions
    tabX = np.array(instanceCifDict['_atom_site_fract_x']).astype(float)
    tabY = np.array(instanceCifDict['_atom_site_fract_y']).astype(float)
    tabZ = np.array(instanceCifDict['_atom_site_fract_z']).astype(float)
    tabX = tabX.reshape((1,tabX.shape[0]))
    tabY = tabY.reshape((1,tabY.shape[0]))
    tabZ = tabZ.reshape((1,tabZ.shape[0]))
    in_positions = np.concatenate((tabX, tabY, tabZ),axis = 0).transpose()
    #Construction maille avec ase
    maille = Atoms(cell = in_cell,
                       numbers = in_numbers,
                       positions = np.zeros(in_positions.shape),
                       pbc=[True, True, True])
    maille.set_scaled_positions(in_positions)
    return maille


def computeSoapReduced(maille):
    '''
    Parameters
    ----------
    maille : ase.Atoms
        La maille associée au cif au format ase.Atoms.

    Returns
    -------
    soap_maille : scipy.sparse.coo_matrix
        Tableau numpy de shape (1, nbFeatures) transformé en sparse Matrix.
        Avec nbFeatures qui dépend de la construction de notre instance
        de SOAP.

    '''
    soap_maille = periodic_soap.create(maille)
    nbAtomes = len(instanceCifDict['_atom_site_fract_x'])
    soap_maille = soap_maille.sum(axis=0).reshape(1, -1) / nbAtomes
    #soap_maille = scipy.sparse.coo_matrix(soap_maille)
    return soap_maille



#==========# Sauvegarde données #==========#

print('creation of x and y ...')

instanceCifDict = extractCifDict(0)
maille = computeMaille(instanceCifDict)
x = computeSoapReduced(maille)

tailleRecup = len(baseDonnee['response']) // 30
precisBarreProgr = 1000

print(f'0 / {precisBarreProgr}')

for i in range(1, tailleRecup):
    instanceCifDict = extractCifDict(18)
    maille = computeMaille(instanceCifDict)
    soap_maille = computeSoapReduced(maille)
    #x = scipy.sparse.vstack([x, soap_maille])   #Si on veut utiliser des sparsematrix
    x = np.concatenate([x, soap_maille], axis=0) #Si on veut utiliser des tableaux numpy
    if (i % (tailleRecup // precisBarreProgr) == 0 
        and i // (tailleRecup//precisBarreProgr) <= precisBarreProgr):
        print(f"{i // (tailleRecup//precisBarreProgr)} / {precisBarreProgr}")


# x a shape (len(baseDonnee['response']), nbFeatures)

y = np.array([baseDonnee['response'][i]['formation_energy_per_atom'] 
          for i in range(tailleRecup)])

print('x and y created')

np.savetxt('baseDonneeX2.csv', x, delimiter=",")
np.savetxt('baseDonneeY2.csv', y, delimiter=",")

print('x and y saved')


