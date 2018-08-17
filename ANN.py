from numpy import*
class neuralNetwork(object):
	def __init__(self):
		random.seed(1)

		self.weights = 2*random.random((3,1))-1

	def __sigmoid(self,x):
		return 1/(1+exp(-x))

	def __derivative_sigmoid(self,x):
		return x*(1-x)

	def train(self,inputs,outputs,iterations):
		for iteration in range(iterations):
			output=self.learn(inputs)
			err=outputs-output
			factor = dot(inputs.T,err*self.__derivative_sigmoid(output))
			self.weights+=factor

	def learn(self,inputs):
		return self.__sigmoid(dot(inputs,self.weights))

if __name__ == "__main__":
 
    #Initialize
    neural_network = neuralNetwork()
 
    # The training set.
    inputs = array([[0, 1, 0], [1, 0, 0], [0 ,1, 1]])
    outputs = array([[0, 0, 1]]).T
 
    # Train the neural network
    neural_network.train(inputs, outputs, 10000)
 
    # Test the neural network with a test example.
    print( neural_network.learn(array([1, 1, 1])))


