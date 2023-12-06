print("Bon courage pour la suite")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from sklearn.preprocessing import normalize

#from sklearn.cluster import KMeans
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.model_selection import train_test_split, GridSearchCV

#from sklearn.metrics import accuracy_score, confusion_matrix
#from sklearn.metrics import recall_score, f1_score


carac = pd.read_csv("C:/Users/emaillot/Documents/GitHub/iut_sd3_accidents/data/carac.csv",sep=';')
lieux = pd.read_csv("C:/Users/emaillot/Documents/GitHub/iut_sd3_accidents/data/lieux.csv",sep=';')
veh = pd.read_csv("C:/Users/emaillot/Documents/GitHub/iut_sd3_accidents/data/veh.csv",sep=';')
vict = pd.read_csv("C:/Users/emaillot/Documents/GitHub/iut_sd3_accidents/data/vict.csv",sep=';')

victime = vict.merge(veh,on=['Num_Acc','num_veh'])
accident = carac.merge(lieux,on = 'Num_Acc')
victime = victime.merge(accident,on='Num_Acc')

nan_values = victime.isna().sum()

nan_values = nan_values.sort_values(ascending=True)*100/127951






ax = nan_values.plot(kind='barh', 
                     figsize=(8, 10), 
                     color='#AF7AC5',
                     zorder=2,
                     width=0.85)

ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(axis="both", 
               which="both", 
               bottom="off", 
               top="off", 
               labelbottom="on", 
               left="off", 
               right="off", 
               labelleft="on")

vals = ax.get_xticks()

for tick in vals:
  ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
plt.show()
# ... La suite de votre code pour la personnalisation du graphique ...
# Par exemple, le tracé des lignes verticales avec ax.axvline()...

# Enregistrer le graphique en tant qu'image PNG
#plt.savefig('C:/Users/emaillot/Documents/my_virtual_envs/graphique.png', dpi=300, bbox_inches='tight')


#plt.savefig('graphique.png', dpi=300, bbox_inches='tight')

nans = ['v1','v2','lartpc',
       'larrout','locp','etatp',
       'actp','voie','pr1',
       'pr','place']


# Créer une copie du DataFrame initial
victime_copie = victime.copy()

# Supprimer les lignes avec des valeurs manquantes et obtenir les lignes supprimées
lignes_supprimees = victime_copie.dropna()

# Obtenir les lignes supprimées en inversant le masque booléen
lignes_supprimees = victime_copie[~victime_copie.index.isin(lignes_supprimees.index)]

nans = ['v1','v2','lartpc',
       'larrout','locp','etatp',
       'actp','voie','pr1',
       'pr','place']

victime = victime.drop(columns = nans)
victime = victime.dropna()



victime.to_csv('C:/Users/emaillot/Documents/GitHub/iut_sd3_accidents/data/step1.csv', index=False)



#lignes_supprimees.to_csv('C:/Users/emaillot/Documents/my_virtual_envs/missing_values_deleted.csv', index=False)
