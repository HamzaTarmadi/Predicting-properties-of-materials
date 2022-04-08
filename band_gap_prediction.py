from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition.orbital import AtomicOrbitals
from matminer.featurizers.composition.orbital import ValenceOrbital
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np



print("#################PREPARING DATA...#####################")
# Importer les données qu'on va utiliser pour l'apprentissage
# On prend tous les matériaux disponibles sur MaterialsProject contenant deux atomes
mpdr = MPDataRetrieval(api_key='3g9wMps4VR5Hy6Pm') # api_key correspond à la clé propre au compte crée sur MaterialProjects
df = mpdr.get_dataframe(criteria={"nelements": 2}, properties=['pretty_formula', 'band_gap', 'structure'])

# On ajoute une colonne montrant les éléments contenus dans le métal
df = StrToComposition().featurize_dataframe(df, "pretty_formula")

# On crée de nouvelles colonnes qui contiennent les caractéristiques de chaque atome
atob = AtomicOrbitals() 
vaob = ValenceOrbital()
df = atob.featurize_dataframe(df, "composition")
df = vaob.featurize_dataframe(df, "composition")

# On prépare les données pour l'apprentissage
X, y = df.drop(['pretty_formula','band_gap', 'structure', 'composition'], axis=1), df['band_gap']
print("#################DONE SUCCESSFULLY#####################")


print("#################PREPARING THE LINEAR REGERSSION...#####################")

# On procède à la création du modèle et à l'apprentissage
model = LinearRegression(X, y)

print("#################DONE SUCCESSFULLY#####################")

print(f'training R2 ={str(round(model.score(X, y), 3))}')
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=model.predict(X))))


print("#################TESTING THE MODEL...#####################")

df_test = pdr.get_dataframe(criteria={"nelements": 3}, properties=['pretty_formula', 'band_gap', 'structure'])
df_test = df_test.iloc[:500,:]
df_test = atob.featurize_dataframe(df_test, "composition")
df_test = vaob.featurize_dataframe(df_tet, "composition")
X_test, y_test = df_test.drop(['pretty_formula','band_gap', 'structure', 'composition'], axis=1), df_test['band_gap']

y_test_pred = model.predict(X_test)
print('testing RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred)))


