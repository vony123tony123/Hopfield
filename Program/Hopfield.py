import numpy as np

class Hopfield:

	def train(self, data):
		p = len(data[0])
		N = len(data)
		self.W = np.zeros((p,p))
		for i in range(N):
			self.W += np.dot(np.transpose([data[i]]), [data[i]])
		self.W = np.array(self.W)
		self.W = (self.W - N * np.identity(p))/p
		self.theta = self.W.sum(axis = -1)

	def predict(self, data):
		prview_result = data
		while True:
			u = np.dot(self.W, np.transpose([prview_result])).flatten()
			for i in range(len(u)):
				if u[i] > 0:
					u[i] = 1
				if u[i] < 0:
					u[i] = -1
				if u[i] == 0:
					u[i] = data[i]
			if np.array_equal(u, prview_result):
				break
			prview_result = u
		return u

