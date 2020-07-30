import random as rand

def get_rand(w,h, alpha):
	# print(w, h)
	ratio = w*h/(fw*fh)
	# print(w, h, 1-ratio)
	# print(ratio)
	if 1-ratio > alpha:
		return rand.randint(-5,5)
	return 0
	# if rand.uniform(0,1) > max(1/(i-j), 0.56):
	# 	return rand.randint(-5,5)
	# else:
	# 	return 0

size = 1000
H = 25000
W = 45000
fw = 1920
fh = 1200

p = [0,0]
h = 1
v = 1

points = []
graph = []

for i in range(size):
	graph.append({})
	xd = rand.randint(200,500)
	yd = rand.randint(50,100)
	points.append((p[0], p[1]))
	if h == 1:
		if p[0] + xd + fw> W:
			p[0] = W - fw
			h = -1
		else:
			p[0] += xd
	else:
		if p[0] - xd < 0:
			p[0] = 0
			h = 1
		else:
			p[0] -= xd
	if v == 1:
		if p[1] + yd + fh> H:
			p[1] = H - fh
			v = -1
		else:
			p[1] += yd
	else:
		if p[1] - yd < 0:
			p[1] = 0
			v = 1
		else:
			p[1] -= yd
for alpha in range(80, 100, 1):
	s = 0
	z = 0
	for i in range(1,size):
		# graph[i][i-1] = get_rand(i,i-1)
		for j in range(i):
			if abs(points[i][0] - points[j][0]) < fw and abs(points[i][1] - points[j][1]) < fh:
				graph[i][j] = get_rand(abs(points[i][0] - points[j][0]),abs(points[i][1] - points[j][1]), alpha/100)
				s += 1
				if graph[i][j] == 0:
					z += 1
	sparsity = z/s * 100
	print(sparsity)

	f = open('graphs/graph_' + str(int(sparsity))+ '.txt','w')
	for i in range(size):
		for j in graph[i]:
			f.write(str(i+1) + '->' + str(j+1) + ':' + str(graph[i][j]) + '\n')


