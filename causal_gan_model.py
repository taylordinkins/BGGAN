
# coding: utf-8

import argparse
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)
KEEP_ATTS = ['Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Smiling', 'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Narrow_Eyes']

def create_bayes_net():
	atts = pd.read_csv('./data/list_attr_celeba.csv')
	atts = atts[KEEP_ATTS]
	graph = BayesianModel()
	graph.add_nodes_from(atts.columns)

	# can't automate this part
	# defining the structure of edges
	graph.add_edges_from([('Young', 'Eyeglasses'), ('Young', 'Bald'), ('Young', 'Mustache'), ('Male', 'Mustache'), 
                      ('Male', 'Smiling'), ('Male', 'Wearing_Lipstick'), ('Young', 'Mouth_Slightly_Open'), 
                      ('Young', 'Narrow_Eyes'), ('Male', 'Narrow_Eyes'), ('Smiling', 'Narrow_Eyes'), 
                      ('Smiling', 'Mouth_Slightly_Open'), ('Young', 'Smiling')])

	# fit estimates the CPD tables for the given structure
	graph.fit(atts)

	return graph

def print_cpd_tables(graph):
	for cpd in graph.get_cpds():
		print(cpd)

# targets are nodes we want to return CPD for
# given evidence nodes and values
# targets = list, evidence = dictionary
def graph_inference(graph, targets, evidence):
	inf = VariableElimination(graph)
	query = inf.query(variables=targets, evidence=evidence)
	
	for t in targets:
		print(query[t])
		print(query[t].variables, query[t].values)

	return query

def main():
	graph = create_bayes_net()
	targets = ['Smiling', 'Male']
	evidence = {'Young': 1, 'Mouth_Slightly_Open': 1}
	query = graph_inference(graph, targets, evidence)

	# Ideally, we do something with the targets, values from here

main()


