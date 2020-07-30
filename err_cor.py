import re
import numpy as np
import cvxpy as cvx
from os import walk

class Graph():
	def __init__(self):
		self.g = {}

	def add_edge(self, edge):
		if edge[0] not in self.g:
			self.g[edge[0]] = {}
		if edge[1] not in self.g:
			self.g[edge[1]] = {}
		self.g[edge[0]][edge[1]] = [None, edge[2], 0]

	def bfs(self, start, des):
		queue = [(start, [start])]
		visited = []
		while True:
			node = queue[0]
			if node[0] == des:
				break;
			queue = queue[1:]
			for n in self.g[node[0]]:
				if n in visited:
					continue
				queue.append((n, node[1] + [n]))
				visited.append(n)
		return node[1]

graph = None
matrix = None 
edge_prob = []
all_predicted = []
all_real = []
weight_effect = 0.1

def calculate_weight(alpha=0.5, n = 50):
    probs = [alpha]
    w = [[1,0]]
    for i in range(1, n):
        prev_weight = probs[i-1]
        val = alpha*prev_weight + 2*np.exp(-i/alpha**2)
        probs.append(val)
        w.append([alpha*prev_weight/val, alpha*(1-prev_weight)/(1-val)])
    return w

def compressed_sense(equation):
	A0 = np.array(equation[0])
	Y0 = np.array(equation[2])
	W = np.array(equation[3])
	X0 = cvx.Variable(A0.shape[1])
	objective = cvx.Minimize(cvx.norm(X0, 1))
	prob = cvx.Problem(objective, [A0*X0 == Y0])
	result = prob.solve()
	return X0.value

def construct_mat(edges, alpha):
	unknown_edges = []
	weights_dict = {}
	A = []
	Y = []
	for e in edges:
		if e[0]-1 > e[1]:
			path = [e[0]] + graph.bfs(e[0]-1, e[1])
			row = [0]*len(unknown_edges)
			A.append(row)
			err_sum = 0
			for i in range(len(path)-1):
				edge_val = matrix[path[i]][path[i+1]]
				err_sum += edge_val[1]
				found = False
				for u  in range(len(unknown_edges)):
					if unknown_edges[u] == (path[i], path[i+1]):
						row[u] = 1
						found = True
						break
				if not found:
					unknown_edges.append((path[i], path[i+1]))
					row.append(1)
					for r in range(len(A)-1):
						A[r].append(0)

			edge_val = matrix[path[0]][path[len(path)-1]]
			err_sum -= edge_val[1]
			found = False
			for u  in range(len(unknown_edges)):
				if unknown_edges[u] == (path[0], path[len(path)-1]):
					row[u] = -1
					found = True
					break
			if not found:
				unknown_edges.append((path[0], path[len(path)-1]))
				row.append(-1)
				for r in range(len(A)-1):
					A[r].append(0)
			
			for i in range(len(path)-1):
				key = (path[i], path[i+1])
				if key not in weights_dict:
					weights_dict[key] = 1 + weight_effect*(edge_prob[len(path)][1]-alpha)
				if err_sum == 0:
					weights_dict[key] = 1 + weight_effect*(edge_prob[len(path)][0]-alpha)

			key = (path[0], path[len(path)-1])
			if key not in weights_dict:
				weights_dict[key] = 1 + weight_effect*(edge_prob[len(path)][1]-alpha)
			if err_sum == 0:
				weights_dict[key] = 1 + weight_effect*(edge_prob[len(path)][0]-alpha)
		
			Y.append(err_sum)
	
	weights = []
	for i in range(len(unknown_edges)):
		weights.append([0]*len(unknown_edges))
		weights[i][i] = weights_dict[unknown_edges[i]]

	return (A, unknown_edges, Y, weights)

def evaluate(predicted, real):
	global all_n
	global all_propagated

	n = len(predicted)
	zero_n = 0
	eval = {'accuracy':0, 'zero accuracy':0, 'propagated':0, 'absolute propagated':0}

	for i in range(n):
		val = predicted[i] - real[i]
		eval['propagated'] += val
		eval['absolute propagated'] += abs(val)
		if round(val) == 0:
			eval['accuracy'] += 1
		if real[i] == 0:
			zero_n += 1
			if round(val) == 0:
				eval['zero accuracy'] += 1

	eval['propagated'] /= n
	eval['absolute propagated'] /= n
	eval['accuracy'] /= n
	eval['zero accuracy'] /= zero_n

	return eval

def solve_partition(edges, alpha):
	if len(edges) == 0:
		return
	equation = construct_mat(edges, alpha)
	res = compressed_sense(equation)
	predicted = []
	real = []

	for i in range(len(equation[1])):
		edge = matrix[equation[1][i][0]][equation[1][i][1]]
		if edge[0] is None:
			edge[0] = res[i]
		else:
			edge[0] = (edge[2]*edge[0] + res[i])/(edge[2]+1)
		edge[2] += 1
		predicted.append(edge[0])
		real.append(edge[1])

	all_predicted.extend(predicted)
	all_real.extend(real)
		
	# print(evaluate(predicted, real))

def flatten(input_list):
	return_list = list()
	for l in input_list:
		return_list.extend(l)
	return return_list

def main(file_path, alpha):
	global graph
	global matrix
	global edge_prob

	print(file_path)
	PART = 100
	file = open(file_path)
	graph = Graph()
	matrix = graph.g
	edge_prob = calculate_weight(alpha=alpha)
	# all_edges = dict()
	edges = dict()
	for l in file:
		edge = list(map(int, re.split('->|:', l)))
		if edge[0] not in edges:
			if len(edges) == PART:
				solve_partition(flatten(list(edges.values())), alpha)
				edges = dict()
			edges[edge[0]] = []
		edges[edge[0]].append(edge)
		# if edge[1] not in edges:
		# 	for e in range(edge[0]-1, edge[1], -1):
		# 		if e not in edges and e in all_edges:
		# 			edges[e] = all_edges[e]
		# 			if len(edges) == PART:
		# 				break
		# if edge[0] not in all_edges:
		# 	all_edges[edge[0]] = []
		# all_edges[edge[0]].append(edge)
		graph.add_edge(edge)
	solve_partition(flatten(list(edges.values())), alpha)

	print('total: ',evaluate(all_predicted, all_real))

if __name__ == '__main__':
	for (dirpath, dirnames, filenames) in walk('graphs/'):
		filenames.sort()
		for f in filenames:
			if f.endswith('.txt'):
				splitted = f.split('_')
				main('graphs/' + f, int(splitted[1][:2]))
