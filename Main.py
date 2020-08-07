import re
import numpy as np
import cvxpy as cvx
from os import walk
import time

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
				try:
					if n <= queue[0][0] and n >= des:
						queue.insert(0, (n, node[1] + [n]))
					else:
						queue.append((n, node[1] + [n]))
				except:
					queue.append((n, node[1] + [n]))
				visited.append(n)
		return node[1]

graph = None
matrix = None 
edge_prob = list()

WEIGHT_EFFECT = 0.1
PART = 1000

all_predicted = list()

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
	A0 = equation[0]
	Y0 = equation[2]
	W = equation[3]
	X0 = cvx.Variable(A0.shape[1])
	objective = cvx.Minimize(cvx.norm(X0, 1))
	prob = cvx.Problem(objective, [A0*X0 == Y0])
	result = prob.solve()
	return X0.value

def double_mat_size(A):
	result = np.zeros((2*A.shape[0], 2*A.shape[1]))
	result[:A.shape[0],:A.shape[1]] = A
	return result

def construct_mat(edges, alpha):
	unknown_edges = list()
	unknown_edges_dict = dict()
	weights_dict = {}
	A = np.zeros((5*PART, 5*PART))
	Y = []
	row_counter = 0
	for e in edges:
		if e[0]-1 > e[1]:
			path = [e[0]] + graph.bfs(e[0]-1, e[1])
			row = A[row_counter, ]
			row_counter += 1
			err_sum = 0
			for i in range(len(path)-1):
				edge_val = matrix[path[i]][path[i+1]]
				err_sum += edge_val[1]
				path_edge = (path[i], path[i+1])
				
				not_found = path_edge[0] not in unknown_edges_dict
				if not_found or path_edge[1] not in unknown_edges_dict[path_edge[0]]:
					unknown_edges.append(path_edge)
					u = len(unknown_edges)-1
					if u >= A.shape[0]:
						A = double_mat_size(A)
					row = A[row_counter-1, ]
					if not_found:
						unknown_edges_dict[path_edge[0]] = dict()
					unknown_edges_dict[path_edge[0]][path_edge[1]] = u
					row[u] = 1
				else:
					u = unknown_edges_dict[path_edge[0]][path_edge[1]]
					row[u] = 1

			edge_val = matrix[path[0]][path[len(path)-1]]
			err_sum -= edge_val[1]
			path_edge = (path[0], path[len(path)-1])

			not_found = path_edge[0] not in unknown_edges_dict
			if not_found or path_edge[1] not in unknown_edges_dict[path_edge[0]]:
				unknown_edges.append(path_edge)
				u = len(unknown_edges)-1
				if u >= A.shape[0]:
					A = double_mat_size(A)
				row = A[row_counter-1, ]
				if not_found:
						unknown_edges_dict[path_edge[0]] = dict()
				unknown_edges_dict[path_edge[0]][path_edge[1]] = u
				row[u] = -1
			else:
				u = unknown_edges_dict[path_edge]
				row[u] = -1

			# for i in range(len(path)-1):
			# 	key = (path[i], path[i+1])
			# 	if key not in weights_dict:
			# 		weights_dict[key] = 1 + WEIGHT_EFFECT*(edge_prob[len(path)][1]-alpha)
			# 	if err_sum == 0:
			# 		weights_dict[key] = 1 + WEIGHT_EFFECT*(edge_prob[len(path)][0]-alpha)

			# key = (path[0], path[len(path)-1])
			# if key not in weights_dict:
			# 	weights_dict[key] = 1 + WEIGHT_EFFECT*(edge_prob[len(path)][1]-alpha)
			# if err_sum == 0:
			# 	weights_dict[key] = 1 + WEIGHT_EFFECT*(edge_prob[len(path)][0]-alpha)
		
			Y.append(err_sum)
	A = A[:row_counter, ]
	A = A.T[:len(unknown_edges), ].T
	Y = np.array(Y)

	weights = []
	# for i in range(len(unknown_edges)):
	# 	weights.append([0]*len(unknown_edges))
	# 	weights[i][i] = weights_dict[unknown_edges[i]]
	# weights = np.array(weights)

	return (A, unknown_edges, Y, weights)

def evaluate(predicted, real):
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
	# timer = time.perf_counter()
	equation = construct_mat(edges, alpha)
	res = compressed_sense(equation)
	# print(time.perf_counter() - timer)
	# predicted = []
	# real = []
	# # s = 0
	# # c = 0
	# for i in range(len(equation[1])):
	# 	edge = matrix[equation[1][i][0]][equation[1][i][1]]
	# 	if edge[0] is None:
	# 		edge[0] = res[i]
	# 	else:
	# 		# s += edge[0]-res[i]
	# 		# c += 1
	# 		edge[0] = (edge[2]*edge[0] + res[i])/(edge[2]+1)
	# 	edge[2] += 1
	# 	predicted.append(edge[0])
	# 	real.append(edge[1])
	# 	if edge not in all_predicted:
	# 		all_predicted.append(edge)
	# print(evaluate(predicted, real))
	# if c != 0:
	# 	print(s/c)

def flatten(input_list):
	return_list = list()
	for l in input_list:
		return_list.extend(l)
	return return_list

def start(file_path, alpha):
	global graph
	global matrix
	global edge_prob
	global all_predicted

	timer = time.perf_counter()
	print(file_path)
	file = open(file_path)
	graph = Graph()
	matrix = graph.g
	all_predicted = list()
	edge_prob = calculate_weight(alpha=alpha)
	edges = dict()
	for l in file:
		edge = list(map(int, re.split('->|:', l)))
		if edge[0] not in edges:
			if len(edges) == PART:
				solve_partition(flatten(list(edges.values())), alpha)
				edges = dict()
			edges[edge[0]] = []
		edges[edge[0]].append(edge)
		graph.add_edge(edge)
	solve_partition(flatten(list(edges.values())), alpha)
	print(time.perf_counter()-timer)
	predicted = []
	real = []
	for e in  all_predicted:
		predicted.append(e[0])
		real.append(e[1])
	# print('total: ',evaluate(predicted, real))

if __name__ == '__main__':
	for (dirpath, dirnames, filenames) in walk('graphs/'):
		filenames.sort()
		for f in filenames:
			if f.endswith('.txt'):
				splitted = f.split('_')
				start('graphs/' + f, int(splitted[1][:2])/100)
