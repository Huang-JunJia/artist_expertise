import numpy as np
from tqdm import tqdm
import math
import warnings

import matplotlib.pyplot as plt
import pickle as pkl
from bpr import BPR


def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class VBPR(BPR):

	def __init__(self, max_item, max_user, k=3, lr=0.5, lam_u=0.01, 
				lam_bias=0.01, lam_rated=0.01, lam_unrated=0.01,
				lam_vf = .1, lam_E = .10, lam_vu = .10, lam_dd=0.01,
				n=3, lr2=0.007):


		self.max_item = max_item
		self.max_user = max_user

		self.validation_aucs = []
		self.set_hyperparameters(k, n, lr, lr2, lam_u, lam_bias, lam_rated, lam_unrated,
								lam_vf, lam_E, lam_vu, lam_dd)
		self.__initialize_parameters()


	def set_hyperparameters(self, k, n, lr, lr2, lam_u, lam_bias, 
			lam_rated, lam_unrated, lam_vf, lam_E, lam_vu, lam_dd):

		self.k = k; self.lr = lr; self.lam_u = lam_u; self.lam_bias = lam_bias
		self.lam_rated = lam_rated; self.lam_unrated = lam_unrated
		self.lam_vf = lam_vf; self.lam_E = lam_E; self.lam_vu = lam_vu
		self.n = n; self.lr2 = lr2; self.lam_dd = lam_dd

	def set_max(self, max_item, max_user):
		self.max_item = max_item
		self.max_user = max_user

	def set_visual_data(self, visual_data):
		self.visual_data = visual_data

	def set_dd_dict(self, dd_dict):
		self.dd_dict = dd_dict

	def __initialize_parameters(self):
		self.item_biases = np.random.normal(size=self.max_item)
		self.latent_items = np.random.normal(size=(self.max_item, self.k), scale=0.1)
		self.latent_users = np.random.normal(size=(self.max_user, self.k), scale=0.1)
		self.visual_users = np.random.normal(size=(self.max_user, self.n), scale=0.1)
		self.E = np.random.normal(size=(4096, self.n), scale=0.1)
		self.visual_bias = np.random.normal(size=(4096), scale=0.1)
		self.dd_bias = 0

	def reset_parameters(self):
		self.__initialize_parameters()
		self.validation_aucs = []
	
	def load_parameters(self, filename):
		self.item_biases, self.latent_items, self.latent_users, \
			self.visual_users, self.E, self.visual_bias, self.dd_bias = \
			pkl.load(open(filename, 'rb')) 

	def save_parameters(self, filename):
		pkl.dump((self.item_biases, self.latent_items, self.latent_users, 
			self.visual_users, self.E, self.visual_bias, self.dd_bias),
			open(filename, 'wb'))
	#================ Implementation details ================#

	def BR_opt(self, rated_item_bias, unrated_item_bias, latent_user, 
			latent_rated_item, latent_unrated_item, 
			vf_difference, visual_user, dd_difference):

		bias_difference = rated_item_bias - unrated_item_bias
		latent_difference = np.dot(latent_rated_item - latent_unrated_item, 
									latent_user)

		vf_encoded = np.dot(np.transpose(self.E), vf_difference)
		visual_difference = np.dot(vf_encoded, visual_user)
		visual_bias_difference = np.dot(vf_difference, self.visual_bias)


		return (bias_difference + latent_difference + \
				visual_difference + visual_bias_difference + \
				dd_difference)

	def AUC(self, samples):

		auc = 0.0
		for user, rated_item, unrated_item in tqdm(samples):

			# Extract out the relevant variables
			rated_item_bias = self.item_biases[rated_item]
			unrated_item_bias = self.item_biases[unrated_item]
			latent_user = self.latent_users[user]
			latent_rated_item = self.latent_items[rated_item]
			latent_unrated_item = self.latent_items[unrated_item]

			visual_user = self.visual_users[user]
			rated_vf = self.visual_data[rated_item]
			unrated_vf = self.visual_data[unrated_item]
			vf_difference = rated_vf-unrated_vf            

			dd_difference = self.dd_bias * \
					(self.dd_dict[rated_item]-self.dd_dict[unrated_item])
			# Determine the output, i.e. x_uij 
			br_opt = self.BR_opt(rated_item_bias, unrated_item_bias, 
						latent_user, latent_rated_item, latent_unrated_item,
						vf_difference, visual_user, dd_difference)

			auc += (br_opt > 0)

		return auc/len(samples)


	def train(self, samples, valid, validation_freq=250000):

		count = 0

		lr = self.lr; lam_u = self.lam_u; lam_bias = self.lam_bias
		lam_rated = self.lam_rated; lam_unrated = self.lam_unrated
		lr2 = self.lr2;  lam_vf = self.lam_vf; lam_E = self.lam_E; 
		lam_vu = self.lam_vu; lam_dd = self.lam_dd


		for user, rated_item, unrated_item in tqdm(samples):
    
			# Extract out the relevant variables
			rated_item_bias = self.item_biases[rated_item]
			unrated_item_bias = self.item_biases[unrated_item]
			latent_user = self.latent_users[user]
			latent_rated_item = self.latent_items[rated_item]
			latent_unrated_item = self.latent_items[unrated_item]

			visual_user = self.visual_users[user]
			rated_vf = self.visual_data[rated_item]
			unrated_vf = self.visual_data[unrated_item]
			vf_difference = rated_vf-unrated_vf
			dd_difference = self.dd_bias * \
					(self.dd_dict[rated_item]-self.dd_dict[unrated_item])

			br_opt = self.BR_opt(rated_item_bias, unrated_item_bias, 
					latent_user, latent_rated_item, latent_unrated_item,
					vf_difference, visual_user, dd_difference)

			output = sigmoid(-br_opt)



			# ====== Get Gradients ===========================
			ib_grad = output
			li_grad = output * latent_user
			lu_grad = output * (latent_rated_item-latent_unrated_item)

			E_grad = output * np.dot((vf_difference)[:,None],
									visual_user[None,:])
			vu_grad = output * np.dot(np.transpose(self.E), vf_difference)
			vb_grad = output * (vf_difference)

			dd_grad = output * (self.dd_dict[rated_item]-self.dd_dict[unrated_item])
			# ================================================



			# Perform gradient updates
			self.item_biases[rated_item] = \
				(1-lr*lam_bias) * rated_item_bias + lr*ib_grad

			self.item_biases[unrated_item] = \
				(1-lr*lam_bias) * unrated_item_bias - lr*ib_grad

			self.latent_items[rated_item] = \
				(1-lr*lam_rated) * latent_rated_item + lr*li_grad 

			self.latent_items[unrated_item] = \
				(1-lr*lam_unrated) * latent_unrated_item - lr*li_grad 

			self.latent_users[user] = \
				(1-lr*lam_u) * latent_user + lr*lu_grad

			self.E = (1-lr2*lam_E) * self.E + lr2*E_grad

			self.visual_users[user] = (1-lr2*lam_vu) * visual_user + lr2*vu_grad

			self.visual_bias = (1-lr2*lam_vf)*self.visual_bias + lr2*vb_grad

			self.dd_bias = (1-lr*lam_dd)*self.dd_bias + lr * dd_grad

			count += 1

			if count % validation_freq is 0:
				self.validation_aucs.append(
					self.AUC(valid)
				)
		if len(self.validation_aucs):
			print ('Best accuracy: {}'.format(max(self.validation_aucs)))

	def plot_validation_error(self, validation_freq=250000):
		super().plot_validation_error(validation_freq)

if __name__ == '__main__':
	from data import Data 
	data = Data(False)
	vbpr = VBPR(*data.get_max())
	train_data = data.generate_train_samples(1000000)
	valid_data = data.generate_evaluation_samples(True)
	vbpr.set_visual_data(data.get_visual_data())
	vbpr.set_dd_dict(data.get_dd_dict())
	vbpr.train(train_data, valid_data, 1000000)
	fn = '../cache/VBPR_{}_{}_{}_{}_default_reg.pkl'.format(3, 3, 0.5, 0.007)
	vbpr.save_parameters(fn)
	vbpr.plot_validation_error()