import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

tabTableauPeriodique = ['H', 'He', 'Li', 'Be', 'B', 'C', 
                        'N', 'O', 'F', 'Ne', 'Na', 'Mg', 
                        'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
                        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 
                        'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
                        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 
                        'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
                        'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 
                        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 
                        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
                        'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 
                        'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 
                        'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 
                        'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 
                        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 
                        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 
                        'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 
                        'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 
                        'Mc', 'Lv', 'Ts', 'Og']

#Tableau du nombre de mailles qui contient au moins l'element i, avec i = 0
#pour l'hydrogène
tabNombre = [8118, 8, 19480, 1068, 5430, 7604, 9771, 62068, 9750, 1, 7398,
             8574, 6457, 8969, 13478, 10561, 5643, 2, 6441, 6403, 2063,
             5855, 7601, 5489, 11707, 10386, 9177, 6925, 8021, 5246, 4031,
             4491, 3441, 5512, 2697, 15, 3822, 5298, 3857, 2740, 3453, 4263, 
             649, 2150, 2301, 2751, 3550, 2864, 3604, 4779, 4700, 4217, 2721, 
             147, 3566, 6183, 4377, 2656, 2280, 2577, 515, 2338, 1611, 1447, 
             1742, 1874, 1889, 1930, 1496, 1874, 1465, 1624, 2404, 3722, 1329,
             1110, 1860, 2230, 2343, 2131, 2645, 2634, 3788, 0, 0, 0, 0, 0,
             297, 971, 253, 2071, 342, 389, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#Permet de créer le tableau ci-dessus
'''
tabNombre = []
for i in range(0, 118):
    data = {
        'criteria': {'elements': {'$in': [tabTableauPeriodique[i]]}},
        'properties': [
            'formation_energy_per_atom',
        ]
    }
    r = requests.post('https://materialsproject.org/rest/v2/query',
                     headers={'X-API-KEY': '5RfgLbTbu83tNTjO3Vh4'},
                     data={k: json.dumps(v) for k,v in data.items()})
    response_content = r.json() # a dict
    tabNombre.append(len(response_content['response']))
    print(i, " : ", len(response_content['response']))
'''

def plusCourant(palier):
    tabElem = []
    for i in range(118):
        if tabNombre[i] >= palier:
            tabElem.append(tabTableauPeriodique[i])
    return tabElem

def complementaire(tab, bigTab): #tab inclu dans bigTab
    tabComp = []
    for elem in bigTab:
        if not elem in tab:
            tabComp.append(elem)
    return tabComp

# Le complementaire : sans ça environ 40 000 mailles
# ['He', 'Be', 'Ne', 'Ar', 'Sc', 
# 'Ga', 'Ge', 'As', 'Br', 'Kr', 
# 'Rb', 'Y', 'Zr', 'Nb', 'Mo', 
# 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
# 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 
# 'Cs', 'La', 'Ce', 'Pr', 'Nd', 
# 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
# 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 
# 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 
# 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 
# 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 
# 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 
# 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 
# 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 
# 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 
# 'Mc', 'Lv', 'Ts', 'Og'] 