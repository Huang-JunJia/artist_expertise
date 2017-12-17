import numpy as np
from tqdm import tqdm
import math
import warnings

import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class BPR:

	def __init__(self, max_item, max_user, k=20, lr=0.1, lam_u=0.1, 
				lam_bias=0.1, lam_rated=0.1, lam_unrated=0.1):

		self.max_item = max_item
		self.max_user = max_user

		self.validation_aucs = []
		self.set_hyperparameters(k, lr, lam_u, lam_bias, lam_rated, lam_unrated)
		self.__initialize_parameters()


	def set_hyperparameters(self, k, lr, lam_u, lam_bias, 
			lam_rated, lam_unrated):

		self.k = k; self.lr = lr; self.lam_u = lam_u; self.lam_bias = lam_bias
		self.lam_rated = lam_rated; self.lam_unrated = lam_unrated

	def set_max(self, max_item, max_user):
		self.max_item = max_item
		self.max_user = max_user


	def __initialize_parameters(self):
		self.item_biases = np.random.normal(size=self.max_item)
		self.latent_items = np.random.normal(size=(self.max_item, self.k))
		self.latent_users = np.random.normal(size=(self.max_user, self.k))

	def reset_parameters(self):
		self.__initialize_parameters()
		self.validation_aucs = []

	

	#================ Implementation details ================#

	def BR_opt(self, rated_item_bias, unrated_item_bias, latent_user, 
			latent_rated_item, latent_unrated_item):

		bias_difference = rated_item_bias - unrated_item_bias
		latent_difference = np.dot(latent_rated_item - latent_unrated_item, 
									latent_user)

		return (bias_difference + latent_difference)

	def AUC(self, samples):

		auc = 0.0
		for user, rated_item, unrated_item in tqdm(samples):

			# Extract out the relevant variables
			rated_item_bias = self.item_biases[rated_item]
			unrated_item_bias = self.item_biases[unrated_item]
			latent_user = self.latent_users[user]
			latent_rated_item = self.latent_items[rated_item]
			latent_unrated_item = self.latent_items[unrated_item]

			# Determine the output, i.e. x_uij 
			br_opt = self.BR_opt(rated_item_bias, unrated_item_bias, latent_user,
					latent_rated_item, latent_unrated_item)

			auc += (br_opt > 0)

		return auc/len(samples)


	def train(self, samples, valid, validation_freq=250000):

		count = 0
		lr = self.lr; lam_u = self.lam_u; lam_bias = self.lam_bias
		lam_rated = self.lam_rated; lam_unrated = self.lam_unrated

		for user, rated_item, unrated_item in tqdm(samples):
    
			# Extract out the relevant variables
			rated_item_bias = self.item_biases[rated_item]
			unrated_item_bias = self.item_biases[unrated_item]
			latent_user = self.latent_users[user]
			latent_rated_item = self.latent_items[rated_item]
			latent_unrated_item = self.latent_items[unrated_item]

			br_opt = self.BR_opt(rated_item_bias, unrated_item_bias, 
					latent_user, latent_rated_item, latent_unrated_item)

			output = sigmoid(-br_opt)

			# Perform gradient updates
			self.item_biases[rated_item] = (1-lr*lam_bias) * rated_item_bias \
					+ lr * output
			self.item_biases[unrated_item] = (1-lr*lam_bias) * unrated_item_bias \
					- lr * output

			self.latent_items[rated_item] = (1-lr*lam_rated) * latent_rated_item \
					+ lr * output * latent_user
			self.latent_items[unrated_item] = (1-lr*lam_unrated) * latent_unrated_item \
					- lr * output * latent_user
			
			self.latent_users[user] = (1-lr*lam_unrated) * latent_user \
					+ lr * output * (latent_rated_item-latent_unrated_item)


			count += 1

			if count % validation_freq is 0:
				self.validation_aucs.append(
					self.AUC(valid)
				)

	def plot_validation_error(self, validation_freq=250000):
		plt.plot(np.array(range(len(self.validation_aucs)+1))*validation_freq, 
				[0.5]+self.validation_aucs)
		plt.show()

if __name__ == '__main__':
	from data import Data 
	data = Data(make_even=True, ndivisions=100)
	bpr = BPR(*data.get_max(), k=10)
	for i in range(10):
		train_data = data.generate_train_samples(10000000)
		valid_data = data.generate_evaluation_samples(True)
		bpr.train(train_data, valid_data,5000000)
	bpr.plot_validation_error(5000000)
	import pdb; pdb.set_trace()