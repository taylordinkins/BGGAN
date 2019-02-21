
# coding: utf-8


import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel

pd.set_option('display.max_columns', 500)


atts = pd.read_csv('./data/list_attr_celeba.csv')
atts = atts[['Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Smiling', 'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Narrow_Eyes']]


graph = BayesianModel()
graph.add_nodes_from(atts.columns)

graph.add_edges_from([('Young', 'Eyeglasses'), ('Young', 'Bald'), ('Young', 'Mustache'), ('Male', 'Mustache'), 
                      ('Male', 'Smiling'), ('Male', 'Wearing_Lipstick'), ('Young', 'Mouth_Slightly_Open'), 
                      ('Young', 'Narrow_Eyes'), ('Male', 'Narrow_Eyes'), ('Smiling', 'Narrow_Eyes'), 
                      ('Smiling', 'Mouth_Slightly_Open'), ('Young', 'Smiling')])

graph.fit(atts)

for cpd in graph.get_cpds():
    print(cpd)

