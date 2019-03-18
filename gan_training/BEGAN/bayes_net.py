import argparse
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
import warnings
import random

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)
KEEP_ATTS = ['Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache',
             'Smiling', 'Wearing_Lipstick', 'Mouth_Slightly_Open',
             'Narrow_Eyes']

def create_bayes_net():
	atts = pd.read_csv('../../data/list_attr_celeba.csv')
	atts = atts[KEEP_ATTS]
	graph = BayesianModel()
	graph.add_nodes_from(atts.columns)

	graph.add_edges_from([('Young', 'Eyeglasses'), ('Young', 'Bald'),
                              ('Young', 'Mustache'), ('Male', 'Mustache'), 
                              ('Male', 'Smiling'), ('Male', 'Wearing_Lipstick'),
                              ('Young', 'Mouth_Slightly_Open'), 
                              ('Young', 'Narrow_Eyes'), ('Male', 'Narrow_Eyes'),
                              ('Smiling', 'Narrow_Eyes'), ('Smiling', 'Mouth_Slightly_Open'),
                              ('Young', 'Smiling')])
	graph.fit(atts)
	return graph

def print_cpd_tables(graph):
	for cpd in graph.get_cpds():
		print(cpd)


def graph_inference(graph, targets, evidence):
	inf = VariableElimination(graph)
	query = inf.query(variables=targets, evidence=evidence)
	return query


def random_evidence(size=3):
	evidence_vars = []
	while len(evidence_vars) < size:
		rand_att = KEEP_ATTS[random.randrange(0, len(KEEP_ATTS))]
		if rand_att not in evidence_vars:
			evidence_vars.append(rand_att)
	evidence = {}
	for ev in evidence_vars:
		evidence[ev] = random.randrange(0, 2)
	return evidence


def evidence_query(targets, values):
	evidence = {}
	for i, targ in enumerate(targets):
		evidence[targ] = values[i]
	return evidence


def return_marginals(graph, batch_size, evidence):
    df = pd.DataFrame(columns=KEEP_ATTS)
    targets = []
    for val in KEEP_ATTS:
        if val not in evidence.keys():
            targets.append(val)
    query = graph_inference(graph, targets, evidence)
    for i in range(batch_size):
        for val in KEEP_ATTS:
            if val not in targets:
                df.loc[i, val] = 1
                # print(val, 1)
            else:
                df.loc[i, val] = query[val].values[1]
                # print(val, query[val].values[1])        
    df = df.apply(pd.to_numeric, downcast='float', errors='coerce')
    return df.values


def main():
	graph = create_bayes_net()
	evidence = random_evidence()
	targets = []
	for val in KEEP_ATTS:
		if val not in evidence.keys():
			targets.append(val)
	query = graph_inference(graph, targets, evidence)
	df = pd.DataFrame(columns=KEEP_ATTS)
	for val in KEEP_ATTS:
		if val not in targets:
			df.loc[0, val] = 1
			print(val, 1)
		else:
			df.loc[0, val] = query[val].values[1]
			print(val, query[val].values[1])
	return df


if __name__ == '__main__':
    main()
