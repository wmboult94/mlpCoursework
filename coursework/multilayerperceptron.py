import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform

import pandas as pd
import sys
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""
Basic Multilayer Perceptron implementation
Multiplication of whole arrays rather than using for loops
Warren Boult, University of Sussex, 2018.
"""

# hidden_layout will be a list, eg [10,20], with number of values specifying
# number of hidden layers, and values specifying number of nodes in that layout.

# TODO: adapt implementation to use tensorflow
# TODO: potentially create a sepate regressor class, overriding the init method
class MLPNetwork(BaseEstimator):
	def __init__(self,num_hidden=[1],num_outputs=1,activationFn='sigmoid',outputActivationFn='linear',alpha=0,lambd=0,learning_rate=0.01,tolerance=1e-8,verbose=0,num_epochs=5000):
		# Initialise network params
		self.num_hidden = num_hidden
		self.num_outputs = num_outputs
		self.learning_rate = learning_rate
		self.alpha = alpha
		self.lambd = lambd
		self.num_epochs = num_epochs
		self.tolerance = tolerance
		self.verbose = verbose

		self.activationFn = activationFn
		self.outputActivationFn = outputActivationFn

		# # Temp variables to make work with sklearn, add 1 for bias
		# num_inputs_b = self.num_inputs + 1
		# num_hidden_b = [val + 1 for val in self.num_hidden]
		# # # TODO: change activation function format to work with sklearn stuff
		# # # Initialise activation functions
		# # # Choose which activation function to use for the hidden and output layers
		# # activations = {'sigmoid':sigmoid,'linear':linear,'tanh':tanh,'relu':relu}
		# # activationDerivatives = {'sigmoid':sigmoidDerivative,'linear':linearDerivative,
		# # 							'tanh':tanhDerivative,'relu':reluDerivative}
		# #
		#
		# self.activationFn = activationFn
		# self.outputActivationFn = outputActivationFn
		#
		# # Initialise node activations
		# self.input_activations = np.array([1.0]*num_inputs_b)
		# # self.hidden_activations = np.array([1.0]*self.num_hidden)
		# self.hidden_activations = []
		# for val in num_hidden_b: # multilayer attempts
		# 	self.hidden_activations.append(np.array([1.0]*val))
		# self.output_activations = np.array([1.0]*self.num_outputs)
		#
		# # # He initialisation
		# # hidden_r = np.sqrt(12 / (num_inputs + num_outputs))
		# # output_r =  np.sqrt(12 / (num_hidden + 1))
		# #
		# # self.hidden_weights = np.random.uniform(-hidden_r,hidden_r,(self.num_hidden,self.num_inputs))
		# # self.output_weights = np.random.uniform(-output_r,output_r,(self.num_outputs,self.num_hidden))
		# # self.output_weights = np.random.normal(0,0.01,(self.num_outputs,self.num_hidden))
		#
		# # Initialise weights using inverse square root approach
		# input_std = 1.0 / np.sqrt(num_inputs_b)
		# hidden_stds = [1.0 / np.sqrt(val) for val in num_hidden_b]
		#
		# # self.hidden_weights = np.random.normal(0,,hidden_std,(self.num_hidden,self.num_inputs))
		# # self.output_weights = np.random.normal(0,output_std,(self.num_outputs,self.num_hidden[-1]))
		# # self.hidden_weights = np.random.normal(0,0.01,(self.num_hidden,self.num_inputs))
		#
		# ####### TODO: finish off from here multilayer
		# # print(self.num_hidden[-1])
		# self.hidden_weights = []
		# self.output_weights = np.random.normal(0,hidden_stds[-1],(self.num_outputs,num_hidden_b[-1])) # multilayer attempt
		# for l, val in enumerate(num_hidden_b): # multilayer attempts
		# 	if l == 0:
		# 		self.hidden_weights.append(np.random.normal(0,0.01,(val,num_inputs_b)))
		# 	else:
		# 		self.hidden_weights.append(np.random.normal(0,0.01,(val,num_hidden_b[l-1])))
		# #######
		#
		# # print(np.array(self.hidden_weights)[0].shape)
		# # sys.exit(0)
		# self.old_output_weights = self.output_weights
		# self.old_hidden_weights = self.hidden_weights

	# Forward pass through network
	def _feedForward(self,inputs):
		if len(inputs) != self.num_inputs_:
			raise ValueError("Input pattern length does not match required length")

		############### Set activation function to be used #####################
		activations = {'sigmoid':sigmoid,'linear':linear,'tanh':tanh,'relu':relu}
		if self.activationFn not in activations.keys():
			raise ValueError('Error: incorrect input activation function specified\nPlease choose from: sigmoid, tanh, linear, relu')
		if self.outputActivationFn not in activations.keys():
			raise ValueError('Error: incorrect output activation function specified\nPlease choose from: sigmoid, tanh, linear, relu')

		activationFn = activations[self.activationFn]
		outputActivationFn = activations[self.outputActivationFn]
		########################################################################

		############### Read in input, ##################
		## final activation value is bias for input and hidden layer
		for i in range(self.num_inputs_):
			self.input_activations_[i] = inputs[i]

		for l, weights in enumerate(self.hidden_weights_):
			# print(weights.shape)
			# print(weights)
			if l == 0:
				self.hidden_activations_[l] = activationFn(np.dot(weights,self.input_activations_))
			else:
				self.hidden_activations_[l] = activationFn(np.dot(weights,self.hidden_activations_[l-1]))
		self.output_activations_ = outputActivationFn(np.dot(self.output_weights_,self.hidden_activations_[-1]))
		########################################################################

		# print(self.output_activations[:])
		return self.output_activations_[:]

	# Perform back-propagation of weights through gradient descent
	def _backPropagate(self,target):

		activationDerivatives = {'sigmoid':sigmoidDerivative,'linear':linearDerivative,
									'tanh':tanhDerivative,'relu':reluDerivative}

		activationDerivative = activationDerivatives[self.activationFn]
		outputActivationDerivative = activationDerivatives[self.outputActivationFn]

		# Initialise values
		delta_output = [0.0] * self.num_outputs
		delta_hidden = [[0.0] * val for val in self.num_hidden]
		target = [target]

		######################## Calculate deltas ##############################
		delta_output = -outputActivationDerivative(self.output_activations_) * (target - self.output_activations_)
		for l, activation in enumerate(self.hidden_activations_):
			if l == len(self.hidden_activations_)-1:
				delta_hidden[l] = activationDerivative(activation) * np.sum((self.output_weights_.T * delta_output).T, axis=0)
			else:
				delta_hidden[l] = activationDerivative(activation) * np.sum((self.hidden_weights_[l+1].T * delta_hidden[l+1]).T, axis=0)
		########################################################################

		######################## Calculate gradients ###########################
		# Outer product multiplies delta_output[k] with each hidden_activation,
		# so that delta_output[k]*hidden_activations[j] goes in entry [k][j]
		gradient = np.outer(delta_output, self.hidden_activations_[-1])
		if self.lambd: # Regularisation
			gradient += self.lambd * self.output_weights_
		new_output_weights = self.output_weights_ - self.learning_rate * gradient
		if self.alpha: # Momentum
			new_output_weights += self.alpha * (self.output_weights_ - self.old_output_weights_)

		new_hidden_weights = []
		for l, weights in enumerate(self.hidden_weights_):
			if l == 0:
				gradient = np.outer(delta_hidden[l], self.input_activations_)
			else:
				gradient = np.outer(delta_hidden[l], self.hidden_activations_[l-1])
			if self.lambd: # Regularisation
				gradient += self.lambd * weights
			temp_hidden_weights = weights - self.learning_rate * gradient
			if self.alpha: # Momentum
				temp_hidden_weights += self.alpha * (weights - self.old_hidden_weights_[l])
			new_hidden_weights.append(temp_hidden_weights)
		########################################################################

		# ## Calculate deltas ##
		# delta_output = -self.outputActivationDerivative(self.output_activations) * (target - self.output_activations)
		# delta_hidden = self.activationDerivative(self.hidden_activations) * np.sum((self.output_weights.T * delta_output).T, axis=0)
		#
		# ## Calculate gradients ##
		# # Outer product multiplies delta_output[k] with each hidden_activation,
		# # so that delta_output[k]*hidden_activations[j] goes in entry [k][j]
		# gradient = np.outer(delta_output, self.hidden_activations)
		# if self.lambd: # Regularisation
		# 	gradient += self.lambd * self.output_weights
		# new_output_weights = self.output_weights - self.learning_rate * gradient
		# if self.alpha: # Momentum
		# 	new_output_weights += self.alpha * (self.output_weights - self.old_output_weights)
		#
		# gradient = np.outer(delta_hidden, self.input_activations)
		# if self.lambd: # Regularisation
		# 	gradient += self.lambd * self.hidden_weights
		# new_hidden_weights = self.hidden_weights - self.learning_rate * gradient
		# if self.alpha: # Momentum
		# 	new_hidden_weights += self.alpha * (self.hidden_weights - self.old_hidden_weights)

		######################### Set new weights ##############################
		self.old_hidden_weights_ = self.hidden_weights_
		self.old_output_weights_ = self.output_weights_
		self.hidden_weights_ = new_hidden_weights
		self.output_weights_ = new_output_weights
		# print(len(self.hidden_weights_))

		# print(self.hidden_weights_)
		########################################################################

		# Calculate current mean squared error loss
		error = np.square(target - self.output_activations_).mean()
		# print('Current use for error: ', error)
		# print('potential use for error: ', target-self.output_activations)
		# error = np.sum(abs(target - self.output_activations))
		# print(target - self.output_activations)

		return error

	# Train network on input data
	def fit(self,X,y):
		print('> Starting training of network...')
		self.num_inputs_ = X.shape[1]
		print('\n* Num inputs: ', self.num_inputs_)

		################## Initialise activations and weights ##################
		# Temp variables to make work with sklearn, add 1 for bias
		num_inputs_b = self.num_inputs_ + 1
		num_hidden_b = [val + 1 for val in self.num_hidden]

		# Initialise node activations
		self.input_activations_ = np.array([1.0]*num_inputs_b)
		# self.hidden_activations = np.array([1.0]*self.num_hidden)
		self.hidden_activations_ = []
		for val in num_hidden_b: # multilayer attempts
			self.hidden_activations_.append(np.array([1.0]*val))
		self.output_activations_ = np.array([1.0]*self.num_outputs)

		# # He initialisation
		# hidden_r = np.sqrt(12 / (num_inputs + num_outputs))
		# output_r =  np.sqrt(12 / (num_hidden + 1))
		#
		# self.hidden_weights = np.random.uniform(-hidden_r,hidden_r,(self.num_hidden,self.num_inputs))
		# self.output_weights = np.random.uniform(-output_r,output_r,(self.num_outputs,self.num_hidden))
		# self.output_weights = np.random.normal(0,0.01,(self.num_outputs,self.num_hidden))

		# Initialise weights using inverse square root approach
		input_std = 1.0 / np.sqrt(num_inputs_b)
		hidden_stds = [1.0 / np.sqrt(val) for val in num_hidden_b]

		# print(self.num_hidden[-1])
		self.hidden_weights_ = []
		self.output_weights_ = np.random.normal(0,hidden_stds[-1],(self.num_outputs,num_hidden_b[-1])) # multilayer attempt
		for l, val in enumerate(num_hidden_b): # multilayer attempts
			if l == 0:
				self.hidden_weights_.append(np.random.normal(0,0.01,(val,num_inputs_b)))
			else:
				self.hidden_weights_.append(np.random.normal(0,0.01,(val,num_hidden_b[l-1])))

		self.old_output_weights_ = self.output_weights_
		self.old_hidden_weights_ = self.hidden_weights_
		########################################################################

		################### Perform training of network ########################
		self.errors_ = []
		error = 0.0
		for i in range(self.num_epochs):
			# print('epoch num: ', i)
			X, y = shuffle(X,y) # randomise order of samples
			for j,sample in enumerate(X):
				# print(pattern)
				target = y[j]
				self._feedForward(sample)
				error += self._backPropagate(target)
			error = error/len(X)
			(self.errors_).append(error)

			# if i != 0 and i % 100 == 0:
			if self.verbose:
				print("> Average error after %d epochs: %f" % (i, error))
				# print("Current output weights: ", self.output_weights)
				# np.random.shuffle(cv_data)
				predictions = self.predict(X[-10:])
				cv_targets = y[-10:]
				print('CV error: ', np.square(cv_targets-predictions).mean())
				print('\nPredictions: ', predictions)
				print('\nTargets: ', cv_targets)

			if i > 10:
				if abs(error - np.average(self.errors_[-3:-1])) < self.tolerance:
					print('> Error not improving, finishing')
					return self

				if i % 20 == 0:
					# print(np.average(errors[-100:]))
					if (error - max(self.errors_[-20:-1])) >= -0.1:
						self.learning_rate = max(5e-06,0.9*self.learning_rate)
						if self.verbose:
							print('\n** Current learning rate: %f \n' % self.learning_rate)

			# 	if error <= np.average(errors[-10:]):
			# 		self.learning_rate = 0.75*self.learning_rate
			# 		# self.alpha = 1.1*self.alpha
			# 	else:
			# 		self.learning_rate = 1.1*self.learning_rate
			#
			# 	# print([val[1] for val in data[:10]])
			#
			# 	# print("Current output activations: ", self.output_activations)
			#
			# if i != 0 and i % 1000 == 0:
			# 	flag = 1
			# 	self.alpha = 0.9*self.alpha
		########################################################################

		self.coef_ = np.average(self.hidden_weights_[0],axis=0)[:-1]
		# print('Coefs: ', self.coef_)

		return self

	# Make predictions on input data
	def predict(self,X,y=None):
		predictions = np.zeros((len(X),self.num_outputs))
		for index,sample in enumerate(X):
			# print(sample)
			# print(pattern)
			self._feedForward(sample)
			predictions[index,:] = self.output_activations_

		return predictions

	def score(self, X, y=None):
		# counts number of values bigger than mean
		return(np.square(self.predict(X)-y).mean())

# Sigmoid activation function
def sigmoid(activation):
	return 1/(1+np.exp(-activation))

# Calculate sigmoid derivative of a sigmoid output
# dy/dx = y * (1-y)
def sigmoidDerivative(sigmoid_output):
	return sigmoid_output * (1-sigmoid_output)

# Linear activation function f(x) = x; for regression
def linear(activation):
	return activation

def linearDerivative(linear_output):
	return 1

# Hyperbolic tangent activation function
def tanh(activation):
	return np.tanh(activation)

# Calculate tanh derivative of a tanh output
# dy/dx = 1 - y**2
def tanhDerivative(tanh_output):
	return 1 - tanh_output**2

# ReLu activation function
def relu(activation):
	activation = np.array(activation)
	activation[activation<0] = 0.0
	# print(activation)
	return activation

# Derivative of relu:
# If x <= 0, f'(x)=0, else f'(x)=1
# Therefore if input of ReLu was <=0,
# its input to reluDerivative would be 0
def reluDerivative(relu_output):
	relu_output = np.array(relu_output)
	relu_output[relu_output <= 0] = 0.0
	relu_output[relu_output>0] = 1.0
	return relu_output

def dummyRegressionData():
	x1 = np.linspace(1,100)
	x2 = np.linspace(20,50)
	y = np.array(3.25*x1 + 2*x2 + 12)
	x_vals = np.column_stack((x1,x2))
	x_vals = (x_vals - np.average(x_vals,axis=0))/np.std(x_vals,axis=0)
	final_practice_data = list(zip(x_vals,y))
	return final_practice_data

def dummyXORData():
	# Generate xor data
	data = np.random.uniform(size=(150,2))
	target = [np.logical_xor(x1,x2) for x1,x2 in np.round_(data)]
	# one_hot_target = np.array([[0,1] if val == True else [1,0] for val in target])
	target = np.array([1 if val == True else 0 for val in target])
	final_data = list(zip(data,target))
	# print(final_data)

	# train_data = final_data[:100]
	# test_data = final_data[100:]
	# train_target = target[:100]
	# test_target = target[100:]
	return final_data

def PolynomialKernel(X1, X2, c=0, d=2):
	return (X1*X2 + c)**d

def RBFKernel(X1,X2,sigma=0.1):
	X = np.column_stack((X1,X2))
	X_norm = np.linalg.norm(X, axis=1)**2
	K = np.exp(-X_norm/2*sigma**2)
	return K

def kernelScatterPlot(data,targets,featureInd):
	fig, axs = plt.subplots(data.shape[1])
	for i,ax in enumerate(axs):
		print(i)
		# new_column = PolynomialKernel(cw_data[:,i],cw_data[:,9],d=1)
		new_column = RBFKernel(data[:,i],data[:,featureInd-1])
		# ax.scatter(temp_data[:,i],cw_targets)
		ax.scatter(new_column,targets)
	plt.show()
	sys.exit(0)

def histPlotFeatures(X):
	fig, axs = plt.subplots(X.shape[1])
	# X = np.log(X)
	if X.shape[1] == 1:
		axs.hist(X1)
	else:
		for i in range(X.shape[1]):
			X1 = X[:,i]
			axs[i].hist(X1,20)
	plt.title('Feature histograms')
	plt.show()

def plotFeatures(X):
	for i in range(X.shape[1]):
		X1 = X[:,i]
		rest_data = np.delete(X,i,axis=1)
		fig, axs = plt.subplots(rest_data.shape[1])
		# print(rest_data.T.shape)
		if rest_data.shape[1] == 1:
			axs.scatter(X1,rest_data)
		else:
			for j,X2 in enumerate(rest_data.T):
				print(j)
				print(X2)
				axs[j].scatter(X1,X2)
		plt.show()

def plotTargetVsFeatures(data,targets):
	fig, axs = plt.subplots(data.shape[1])
	if data.shape[1] == 1:
		axs.scatter(data,targets)
	else:
		for i,ax in enumerate(axs):
			print(i)
			ax.scatter(data[:,i],targets)
	plt.xlabel('Features vs targets')
	plt.show()

# Returns basic normalised data, zero mean unit variance
# Data in the form (n_samples,n_features)
def standardize(data):
	return (data - np.mean(data,axis=0))/np.std(data,axis=0)

# Returns min max scaled data
# Data in the form (n_samples,n_features)
def minMaxScale(data):
	return (data - np.min(data,axis=0))/(np.max(data,axis=0) - np.min(data,axis=0))

def PCATransform(data, targets, num_components, normalise=False, plot=False):
	pca = PCA(n_components=num_components)
	if normalise:
		data = standardize(data)
	pca_data = pca.fit_transform(data)
	pca_var = pca.explained_variance_
	for k,comp in enumerate(pca_var):
		print('Percentage variance kept up to component %.0f: %f' % (k+1, sum(pca_var[:k])/sum(pca_var)*100))
	if plot:
		plotTargetVsFeatures(pca_data,targets)
		# histPlotFeatures(pca_data)

	return pca_data

def randomForestRegressor(data,targets,cv_data,cv_targets):
	# RandomForestRegressor test
	regr = RandomForestRegressor(max_depth=3, random_state=0)
	regr.fit(data, targets)
	print( '>   ', regr.predict(cv_data) )
	print('\n\n')
	print(cv_targets)

def sklearnMLPRegressor(data,targets,cv_data,cv_targets,iters):
	# RandomForestRegressor test
	print('Sklearn mlp regressor training...')
	regr = MLPRegressor((16,8),max_iter=iters,alpha=0.0001,learning_rate='adaptive',verbose=True,tol=1e-5)
	regr.fit(data, targets)
	print('Final loss score: ', regr.loss_)
	print( '> Predictions  \n', regr.predict(cv_data) )
	print('\n> Target Values   \n')
	print(cv_targets)

if __name__ == "__main__":

	### DATA GENERATION ###

	# xor_data = dummyXORData()

	# training_set = [train_data,train_target]
	# print('\n\n',train_data)

	# Load coursework data
	cw_data_init = np.loadtxt('data186959.csv',delimiter=',')
	# np.random.shuffle(cw_data_init)
	cw_targets = cw_data_init[:,-1]
	cw_data = cw_data_init[:,:-1]

	########################################

	### DATA PREPROCESSING ###
	max_target = max(cw_targets)
	cw_targets = 100*(cw_targets / max_target)
	# print(np.where(cw_data[:,6]==np.unique(cw_data[:,6])))
	# sys.exit(0)
	# cw_targets = cw_targets/max(cw_targets)*100
	# plt.scatter(cw_data[:,5],cw_targets,s=5)
	# plt.show()
	# plotFeatures(cw_data)
	# fig, ax = plt.subplots(2)
	# ax[0].hist(cw_targets)
	# ax[1].hist(np.log(cw_targets))
	# plt.show()

	new_cw_data = np.column_stack((cw_data[:,0],cw_data[:,1],cw_data[:,8]))
	# plotTargetVsFeatures(p_data,cw_targets)

	# kernelScatterPlot(cw_data,cw_targets,7)
	# new_column = PolynomialKernel(cw_data[:,8],cw_data[:,6])
	new_column28 = RBFKernel(cw_data[:,2],cw_data[:,8])
	new_column67 = RBFKernel(cw_data[:,6],cw_data[:,7])
	temp_data = np.column_stack((new_cw_data,new_column67,new_column28))
	# print(temp_data.shape)
	# pca = PCA(n_components=2)
	# norm_data = StandardScaler().fit(new_cw_data)
	norm_data = StandardScaler().fit(cw_data)
	# norm_data = StandardScaler().fit(temp_data)
	# minmax_data = MinMaxScaler().fit(new_cw_data)
	# histPlotFeatures(minmax_data)
	# sys.exit(0)
	# new_norm_data = StandardScaler().fit(temp_data)
	# pca_data = pca.fit_transform(cw_data)
	# pca_data = PCAPlotter(cw_data,2)
	# sys.exit(0)
	# temp_data = np.column_stack((pca_data,new_column28,new_column66))

	# pca_data = PCATransform(norm_data,cw_targets,4,True)
	# temp_data = np.column_stack((pca_data,StandardScaler().fit(cw_data)[:,8]))
	# sys.exit(0)
	# new_cw_data = np.delete(cw_data,3,1)
	# pcaWhiten = PCA(n_components=9,whiten=True)
	# whitened_data = pcaWhiten.fit_transform(new_cw_data)
	# print(whitened_data)
	# sys.exit(0)
	# pca_var = pcaWhiten.explained_variance_
	# whitened_data = whitened_data / pca_var
	# print(np.var(whitened_data,axis=0))
	# print(pca_var)
	# print(cw_data[:10])
	# print(whitened_data[:10])
	# print(pca_data.shape)
	# practice_targets = cw_targets[:1000]
	# print(practice_targets)
	# practice_data = whitened_data[:1000,:]
	# practice_data = new_cw_data[:1000,:]
	# print(practice_data.shape)
	# practice_data = cw_data[:1000,:]
	practice_data = norm_data[:2000,:]
	# print(practice_data)
	# sys.exit(0)
	practice_targets = cw_targets[:2000]
	# practice_data = temp_data[:1000,:]
	# cv_data = new_cw_data[-100:,:]
	cv_data = norm_data[-100:,:]
	# cv_data = temp_data[-100:,:]
	# cv_data = minmax_data[-100:,:]
	cv_targets = cw_targets[-100:]
	# cv_data = cw_data[-100:,:]
	# cv_data = whitened_data[-10:,:]

	# df = pd.DataFrame(final_practice_data,columns=['Features','Target'])
	# print(df['Target'])
	# final_practice_data = list(zip(random_data,practice_targets))

	#####################################

	### MODEL BUILDING ###

	epochs = 10000
	mlp_regressor = MLPNetwork([60],1,alpha=0.9,lambd=0.0001,learning_rate=0.001,num_epochs=epochs,activationFn='sigmoid',outputActivationFn='linear',verbose=0)
	selector = RFECV(mlp_regressor, step=1, n_jobs=2, scoring='neg_mean_squared_error')
	selector = selector.fit(practice_data, practice_targets)
	print(selector.support_)
	sys.exit(0)
	reg = mlp_regressor.fit(practice_data,practice_targets)
	errors = reg.errors_
	predictions = max_target * (mlp_regressor.predict(cv_data) / 100)
	cv_targets =  max_target * (cv_targets / 100)
	error_score = np.square(predictions-cv_targets).mean()
	print('\nError score: ', error_score)
	print('\nPredictions: ', predictions)
	print('\nTargets: ', cv_targets)
	norm_scaler = StandardScaler()

	pipeline = Pipeline([
		('standardize', norm_scaler),
		('feature_selection', selector),
		('regression', mlp_regressor)
	])

	# mlp = MLPNetwork(2,4,1,learning_rate=0.1,num_epochs=epochs,outputActivationFn='sigmoid')
	# errors = mlp.train(train_data)
	# predictions = mlp.predict(test_data)
	# # print(predictions)
	# # predictions = np.argmax(predictions,axis=1)
	# predictions = np.array([1 if val >= 0.5 else 0 for val in predictions])
	# # print(predictions)
	# # print(target[100:])
	# print('Correct predictions: %.0f / %.0f: ' % (len(np.where(predictions==np.array(target[100:]))[0]), len(target[100:])))
	plt.plot(range(len(errors[10:])),errors[10:])
	plt.show()
	# print(mlp.output_activations)

	# randomForestRegressor(practice_data,practice_targets,cv_data,cv_targets)
	# sklearnMLPRegressor(practice_data,practice_targets,cv_data,cv_targets,iters=20000)

	#######################################

	### MODEL EVALUATION ###


	#####################################
