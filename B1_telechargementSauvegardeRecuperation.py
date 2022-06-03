'''
---------- A Lire ----------

Ce fichier permet de télécharger des données depuis materialProject et de les
sauvegarder au format json.
'''

import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# [Pour télécharger les données depuis le site de MaterialsProject]
data = {
    'criteria': {
        'elements': {'$nin': ['He', 'Be', 'Ne', 'Ar', 
                               'Sc',  'Ga', 'Ge', 'As', 
                               'Br', 'Kr',  'Rb', 'Y', 
                               'Zr', 'Nb', 'Mo',  'Tc', 
                               'Ru', 'Rh', 'Pd', 'Ag', 
                               'Cd',  'In', 'Sn', 'Sb', 
                               'Te', 'I', 'Xe',  'Cs', 
                               'La', 'Ce', 'Pr', 'Nd',  
                               'Pm', 'Sm', 'Eu', 'Gd', 
                               'Tb', 'Dy',  'Ho', 'Er', 
                               'Tm', 'Yb', 'Lu', 'Hf',  
                               'Ta', 'W', 'Re', 'Os', 
                               'Ir', 'Pt',  'Au', 'Hg', 
                               'Tl', 'Pb', 'Bi', 'Po',  
                               'At', 'Rn', 'Fr', 'Ra', 
                               'Ac', 'Th',  'Pa', 'U', 
                               'Np', 'Pu', 'Am', 'Cm',  
                               'Bk', 'Cf', 'Es', 'Fm', 
                               'Md', 'No',  'Lr', 'Rf', 
                               'Db', 'Sg', 'Bh', 'Hs',  
                               'Mt', 'Ds', 'Rg', 'Cn', 
                               'Nh', 'Fl',  'Mc', 'Lv', 
                               'Ts', 'Og']}
    },
    'properties': [
        'cif',
        'formation_energy_per_atom',
        'band_gap'
    ]
}
r = requests.post('https://materialsproject.org/rest/v2/query',
                 headers={'X-API-KEY': '5RfgLbTbu83tNTjO3Vh4'},
                 data={k: json.dumps(v) for k,v in data.items()})
response_content = r.json() # a dict


# [Pour écrire le resultat dans un fichier sous forme de texte]
'''
f = open("dataFormEnCifGrand.txt", "a")
f.write(json.dumps(response_content))
f.close()
'''


# [Pour reprendre les données depuis le fichier et les mettre sous forme de DF]
"""
f = open("dataFormEnCif.txt", "r")
truc = json.loads(f.read())
truc2 = pd.DataFrame(truc['response'])
"""

# Pour récupérer la coordonnée z du deuxième atome dans le cif
"""
from pymatgen.io.cif import CifParser
blurp = CifParser.from_string(truc['response'][0]['cif'])
blurpDict = blurp.as_dict()
list(blurpDict.values())[0]['_atom_site_fract_z'][1]
"""

