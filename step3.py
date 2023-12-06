import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans


victime= pd.read_csv("C:/Users/emaillot/Documents/GitHub/iut_sd3_accidents/data/step1.csv",sep=',',decimal='.')
victime = victime.drop(columns=['an'])
hrmn=pd.cut(victime['hrmn'],24,labels=[str(i) for i in range(0,24)])
victime['hrmn']=hrmn.values

victime.to_csv('C:/Users/emaillot/Documents/my_virtual_envs/time_encoding.csv', index=False)