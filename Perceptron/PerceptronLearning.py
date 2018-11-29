import numpy as np

def perceptronLearner(s, t, alpha = 1, theta = 0.2):
	#s is input training vector NxX
	#t is NxY, so w should be XxY as we will get
	# 1xY output for each training vector
	w = np.zeros((s.shape[1], t.shape[1]))
	b = np.zeros(t.shape[1])
	#y = x.w + b...for each training input
	count = 1
	iterations = 0
	while count!=0:
		count = 0
		for i, si in enumerate(s):
			iterations+=1
			print 'Iteration No:', iterations
			si = np.expand_dims(si, axis = 0)
			ti = np.expand_dims(t[i], axis = 0)
			print 'input vector is:', si
			print 'weight is:', w
			Yprev = si.dot(w) + b
			print 'actual Yprev is:', Yprev
			for yi in range(len(Yprev[0])):
				if Yprev[0][yi]>theta:
					Yprev[0][yi] = 1
				elif Yprev[0][yi]<=theta and Yprev[0][yi]>= -theta:
					Yprev[0][yi] = 0
				else:
					Yprev[0][yi] = -1
			print 'Calculated output:', Yprev,'Expected Output:', ti
			if (Yprev-ti).any():
				'Output is not as expected, so we will update w and b'
				w = w + alpha*si.T.dot(ti)
				print 'delta w is:', si.T.dot(ti)
				b = b + alpha*ti
				print 'delta b is', alpha*ti
				count+=1
			print 'W after this iteration:', w
			print 'b after this iteration:', b

#s = np.array([[1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, -1, -1], [-1, 1, -1, -1], [1, -1, 1, 1], [-1, -1, 1, 1], [1, -1, -1, 1], [-1, -1, -1, 1]])
#s = np.array([[1, 1], [1, 2], [2, -1], [2, 0], [-1, 2], [-2, 1], [-1, -1], [-2, -2]])
#t = np.array([[-1, -1], [-1, -1], [-1, 1], [-1, 1], [1, -1], [1, -1], [1, 1], [1, 1]])
s = np.array([[1, 2], [-1, 2], [0, -1]])
t = np.array([[1], [-1], [-1]])
perceptronLearner(s, t)

