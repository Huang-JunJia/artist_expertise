#==== BUCKETED ARTIST VBPR =======#

import numpy as np
from tqdm import tqdm
import math
import warnings

import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pkl
from avbpr import AVBPR


def sigmoid(x):
	if x < -709:
		sigmoid = 0.0
	else:
		sigmoid = 1.0 / (1.0 + math.exp(-x))
	return sigmoid


class BAVBPR(AVBPR):

	def __init__(self, max_item, max_user, filename, k=3, lr=0.0005, lam_u=0.01, 
				lam_bias=0.01, lam_rated=0.01, lam_unrated=0.01,
				lam_vf = .1, lam_E = .10, lam_vu = .10, lam_dd=0.01,
				n=3, lr2=0.000005, nExpertise=5, lam_smooth=0.1, lam_delta=0.1):

		self.filename = filename 

		self.max_item = max_item
		self.max_user = max_user

		self.validation_aucs = []
		super().set_hyperparameters(k, n, lr, lr2, lam_u, lam_bias, lam_rated, 
								lam_unrated, lam_vf, lam_E, lam_vu, lam_dd, 
								lam_smooth, lam_delta, nExpertise)

		self.__initialize_parameters(filename)


	def __initialize_parameters(self, filename):
		
		self.item_biases, self.latent_items, self.latent_users, \
			self.visual_users, self.E, self.visual_bias, self.dd_bias = \
			pkl.load(open(filename, 'rb')) 

		''' There are two "buckets" for visual parameters. The first one 
			was loaded already. The second one is initialized as a copy
			of the first'''

		self.E2 = self.E[:]
		self.visual_bias2 = self.visual_bias[:]

		self.E3 = self.E[:]
		self.visual_bias3 = self.visual_bias[:]

		'''Now for the expertise parameters'''

		self.E_delta = np.zeros(shape=(4096, self.n, self.nExpertise, 2))
		self.vb_delta = np.zeros(shape=(4096, self.nExpertise, 2))

		'''Per-factor scaling'''
		self.E_w = np.ones(shape=(self.n, self.nExpertise, 2))*0.01
		self.vb_w = np.ones(shape=(4096, self.nExpertise, 2))*0.01

		self.E = np.repeat(np.expand_dims(self.E, axis=2), 2, axis=2)
		self.visual_bias = np.repeat(np.expand_dims(self.visual_bias, axis=1), 2, axis=1)


	def initialize_assignments2(self):
		artist_assignments = {}
		artwork_assignments = {}

		for key in self.artist_dict:
			artworks = self.artist_dict[key]
			nFavs = [self.img_nfavs_dict[x] for x in artworks]
			'''Heuristic for beginners whose artwork don't really improve at all
			# Mean instead of median because in this case, to emphasize these
			# artists generally got no attention one way or another. '''
			if np.mean(nFavs) < 50: 
				artist_assignments[key] = 0
			else:
				'''If the user isn't a total beginner, assume they evolve
				through the expertise levels.'''
				expertise = []; curr = 0
				for artwork in artworks:
					if curr < 1 and self.img_nfavs_dict[artwork] > 150:
						curr = 1

					if curr < 2 and self.img_nfavs_dict[artwork] > 300:
						curr = 2

					if curr < 3 and self.img_nfavs_dict[artwork] > 700:
						curr = 3

					if curr < 4 and self.img_nfavs_dict[artwork] > 1500:
						curr = 4

					expertise.append(curr)
				artist_assignments[key] = dict(zip(artworks, expertise))

		self.artist_assignments = artist_assignments	
		self.artwork_assignments = artwork_assignments

	def initialize_assignments(self):
		artist_assignments = {}
		artwork_assignments = {}

		for key in self.artist_dict:
			artworks = self.artist_dict[key]
			nFavs = [self.img_nfavs_dict[x] for x in artworks]
			'''Heuristic for beginners whose artwork don't really improve at all
			# Mean instead of median because in this case, to emphasize these
			# artists generally got no attention one way or another. '''
			if np.mean(nFavs) < 80: 
				artist_assignments[key] = 0
			elif np.mean(nFavs) > 1000:
				artist_assignments[key] = 1
			else:
				'''If the user isn't a total beginner, assume they evolve
				through the expertise levels.'''
				div = math.ceil(float(len(artworks))/self.nExpertise) 
				expertise = [int(i/div) for i, x in enumerate(artworks)]
				artwork_assignments[key] = dict(zip(artworks, expertise))

				if np.mean(nFavs) < 300:
					artist_assignments[key] = 2
				else:
					artist_assignments[key] = 3
		self.artist_assignments = artist_assignments	
		self.artwork_assignments = artwork_assignments

	#================ Implementation details ================#



	def AUC(self, samples):

		auc = 0.0
		for user, rated_item, unrated_item in tqdm(samples):

			# Extract out the relevant variables
			rated_item_bias = self.item_biases[rated_item]
			unrated_item_bias = self.item_biases[unrated_item]

			latent_user = self.latent_users[user]
			latent_rated_item = self.latent_items[rated_item]
			latent_unrated_item = self.latent_items[unrated_item]

			#Get visual information.
			visual_user = self.visual_users[user]
			rated_vf = self.visual_data[rated_item]
			unrated_vf = self.visual_data[unrated_item]

			#Get who the artist is
			rated_artist = self.item_to_artist[rated_item]
			unrated_artist = self.item_to_artist[unrated_item]

			#Get bucket level
			rated_bucket = self.artist_assignments[rated_artist]
			unrated_bucket = self.artist_assignments[unrated_artist]

			if rated_bucket == 0:
				rated_encoded = np.dot(np.transpose(self.E2), rated_vf)
				rated_visual_bias = self.visual_bias2
			elif rated_bucket == 1:
				rated_encoded = np.dot(np.transpose(self.E3), rated_vf)
				rated_visual_bias = self.visual_bias3

			else:
				if rated_bucket == 2: 
					ind = 0
				else: ind = 1 
				rated_expertise_level = \
					self.artwork_assignments[rated_artist][rated_item]
				rated_E_delta = self.E_delta[:,:,rated_expertise_level,ind]
				rated_E_w = self.E_w[:,rated_expertise_level,ind]
				rated_encoded = np.dot(np.transpose(self.E[:,:,ind]), rated_vf)
				rated_delta_encoded = np.dot(np.transpose(rated_E_delta), rated_vf)
				rated_encoded = np.multiply(rated_E_w, rated_encoded)+\
								rated_delta_encoded

				rated_vb_delta = self.vb_delta[:,rated_expertise_level, ind]
				rated_vb_w = self.vb_w[:, rated_expertise_level, ind]
				rated_visual_bias = rated_vb_delta +\
					np.multiply(self.visual_bias[:,ind], rated_vb_w)

			rated_dd = self.dd_dict[rated_item]*self.dd_bias

			if unrated_bucket == 0:
				unrated_encoded = np.dot(np.transpose(self.E2), unrated_vf)
				unrated_visual_bias = self.visual_bias2
			elif unrated_bucket == 1:
				unrated_encoded = np.dot(np.transpose(self.E3), unrated_vf)
				unrated_visual_bias = self.visual_bias3

			else:
				if unrated_bucket == 2:
					ind = 0
				else:
					ind = 1
				unrated_expertise_level = \
					self.artwork_assignments[unrated_artist][unrated_item]

				unrated_E_delta = self.E_delta[:,:,unrated_expertise_level, ind]
				unrated_E_w = self.E_w[:,unrated_expertise_level,ind]
				unrated_encoded = np.dot(np.transpose(self.E[:,:,ind]), unrated_vf)
				unrated_delta_encoded = np.dot(np.transpose(unrated_E_delta), unrated_vf)
				unrated_encoded = np.multiply(unrated_E_w, unrated_encoded)+\
									unrated_delta_encoded

				unrated_vb_delta = self.vb_delta[:,unrated_expertise_level,ind]
				unrated_vb_w = self.vb_w[:, unrated_expertise_level,ind]
				unrated_visual_bias = unrated_vb_delta +\
					np.multiply(self.visual_bias[:,ind], unrated_vb_w)

			unrated_dd = self.dd_dict[unrated_item]*self.dd_bias

			encoded_difference = rated_encoded-unrated_encoded
			dd_difference = rated_dd-unrated_dd
			# Determine the output, i.e. x_uij 
			br_opt = self.BR_opt(rated_item_bias, unrated_item_bias, latent_user, 
				latent_rated_item, latent_unrated_item, rated_vf, unrated_vf,
				encoded_difference, rated_visual_bias, unrated_visual_bias, 
				visual_user, dd_difference)

			auc += (br_opt > 0)

		return auc/len(samples)



	def train(self, samples, valid, validation_freq=250000):

		count = 0

		lr = self.lr; lam_u = self.lam_u; lam_bias = self.lam_bias
		lam_rated = self.lam_rated; lam_unrated = self.lam_unrated
		lr2 = self.lr2;  lam_vf = self.lam_vf; lam_E = self.lam_E; 
		lam_vu = self.lam_vu; lam_smooth = self.lam_smooth; 
		lam_dd = self.lam_dd; lam_delta = self.lam_delta


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

			rated_artist = self.item_to_artist[rated_item]
			unrated_artist = self.item_to_artist[unrated_item]

			#Get bucket level
			rated_bucket = self.artist_assignments[rated_artist]
			unrated_bucket = self.artist_assignments[unrated_artist]

			if rated_bucket == 0:
				rated_encoded = np.dot(np.transpose(self.E2), rated_vf)
				rated_visual_bias = self.visual_bias2
			elif rated_bucket == 1:
				rated_encoded = np.dot(np.transpose(self.E3), rated_vf)
				rated_visual_bias = self.visual_bias3

			else:
				if rated_bucket == 2: 
					ind = 0
				else: ind = 1 
				rated_expertise_level = \
					self.artwork_assignments[rated_artist][rated_item]
				rated_E_delta = self.E_delta[:,:,rated_expertise_level,ind]
				rated_E_w = self.E_w[:,rated_expertise_level,ind]
				rated_encoded = np.dot(np.transpose(self.E[:,:,ind]), rated_vf)
				rated_delta_encoded = np.dot(np.transpose(rated_E_delta), rated_vf)
				rated_encoded = np.multiply(rated_E_w, rated_encoded)+\
								rated_delta_encoded

				rated_vb_delta = self.vb_delta[:,rated_expertise_level, ind]
				rated_vb_w = self.vb_w[:, rated_expertise_level, ind]
				rated_visual_bias = rated_vb_delta +\
					np.multiply(self.visual_bias[:,ind], rated_vb_w)

			rated_dd = self.dd_dict[rated_item]*self.dd_bias

			if unrated_bucket == 0:
				unrated_encoded = np.dot(np.transpose(self.E2), unrated_vf)
				unrated_visual_bias = self.visual_bias2
			elif unrated_bucket == 1:
				unrated_encoded = np.dot(np.transpose(self.E3), unrated_vf)
				unrated_visual_bias = self.visual_bias3

			else:
				if unrated_bucket == 2:
					ind = 0
				else:
					ind = 1
				unrated_expertise_level = \
					self.artwork_assignments[unrated_artist][unrated_item]

				unrated_E_delta = self.E_delta[:,:,unrated_expertise_level, ind]
				unrated_E_w = self.E_w[:,unrated_expertise_level,ind]
				unrated_encoded = np.dot(np.transpose(self.E[:,:,ind]), unrated_vf)
				unrated_delta_encoded = np.dot(np.transpose(unrated_E_delta), unrated_vf)
				unrated_encoded = np.multiply(unrated_E_w, unrated_encoded)+\
									unrated_delta_encoded

				unrated_vb_delta = self.vb_delta[:,unrated_expertise_level,ind]
				unrated_vb_w = self.vb_w[:, unrated_expertise_level,ind]
				unrated_visual_bias = unrated_vb_delta +\
					np.multiply(self.visual_bias[:,ind], unrated_vb_w)

			unrated_dd = self.dd_dict[unrated_item]*self.dd_bias
			encoded_difference = rated_encoded-unrated_encoded
			dd_difference = rated_dd-unrated_dd
			br_opt = self.BR_opt(rated_item_bias, unrated_item_bias, latent_user, 
				latent_rated_item, latent_unrated_item, rated_vf, unrated_vf,
				encoded_difference, rated_visual_bias, unrated_visual_bias, 
				visual_user, dd_difference)

			output = sigmoid(-br_opt)



			# ====== Get Gradients ===========================
			ib_grad = output
			li_grad = output * latent_user
			lu_grad = output * (latent_rated_item-latent_unrated_item)

			if rated_bucket == 0 or rated_bucket == 1:
				rated_E_grad = output * (np.dot(rated_vf[:,None],visual_user[None,:]))
				rated_vb_grad = output * rated_vf

			else:
				if rated_bucket == 2:
					ind = 0
				else:
					ind = 1

				rated_E_delta_grad = output * (
						np.dot(rated_vf[:,None],visual_user[None,:]))
				rated_E_w_grad = output * np.multiply(rated_encoded, visual_user)
				rated_E_grad = output * np.outer(rated_vf, np.multiply(visual_user, rated_E_w)) 

				rated_vb_delta_grad = output * rated_vf
				rated_vb_w_grad = output * np.multiply(rated_vf, self.visual_bias[:,ind])
				rated_vb_grad = output * np.multiply(rated_vb_w, rated_vf)

			if unrated_bucket == 0 or unrated_bucket == 1:
				unrated_E_grad = -output * (np.dot(rated_vf[:,None],visual_user[None,:]))
				unrated_vb_grad = -output * unrated_vf
			else:
				if unrated_bucket == 2:
					ind = 0
				else:
					ind = 1

				unrated_E_delta_grad = -output * (
					np.dot(unrated_vf[:,None],visual_user[None,:]))
				unrated_E_w_grad = -output * np.multiply(unrated_encoded, visual_user)
				unrated_E_grad = -output* np.outer(unrated_vf, np.multiply(visual_user, unrated_E_w))

				unrated_vb_delta_grad = -output * unrated_vf
				unrated_vb_w_grad = -output * np.multiply(unrated_vf, self.visual_bias[:,ind])
				unrated_vb_grad = -output * np.multiply(unrated_vb_w, unrated_vf)


			rated_dd_grad = output * rated_dd 
			unrated_dd_grad = -output * unrated_dd

			vu_grad = output * (encoded_difference)

			# ================================================


			# Perform gradient updates
			self.item_biases[rated_item] = (1-lr*lam_bias) * rated_item_bias + lr*ib_grad
			self.item_biases[unrated_item] = (1-lr*lam_bias) * unrated_item_bias - lr*ib_grad

			self.latent_items[rated_item] = (1-lr*lam_rated) * latent_rated_item + lr*li_grad 
			self.latent_items[unrated_item] = (1-lr*lam_unrated) * latent_unrated_item - lr*li_grad 

			self.latent_users[user] = (1-lr*lam_unrated) * latent_user + lr*lu_grad



			# Now, update all the rated parameters:

			if rated_bucket == 0:
				self.E2 = (1-lr2*lam_E)*self.E2 + lr2 * rated_E_grad
				self.visual_bias2 = (1-lr2*lam_vf)*self.visual_bias2 + \
					lr2 * rated_vb_grad
			elif rated_bucket == 1:
				self.E3 = (1-lr2*lam_E)*self.E3 + lr2 * rated_E_grad
				self.visual_bias3 = (1-lr2*lam_vf)*self.visual_bias3 + \
					lr2 * rated_vb_grad

			else:
				if rated_bucket == 2: 
					ind = 0
				else: ind = 1 

				'''
				Get the smoothing expertise levels
				'''
				if rated_expertise_level == 0:
					smaller = rated_expertise_level; larger = rated_expertise_level + 1
				elif rated_expertise_level == self.nExpertise-1:
					smaller = rated_expertise_level-1; larger = rated_expertise_level
				else:
					smaller = rated_expertise_level-1; larger = rated_expertise_level+1

				E_delta_smooth = lam_smooth * \
					(self.E_delta[:,:,smaller,ind] - self.E_delta[:,:,larger,ind])

				self.E_delta[:,:,rated_expertise_level,ind] = \
					(1-lr2*lam_E) * self.E_delta[:,:,rated_expertise_level,ind] \
					+ lr2*rated_E_delta_grad - lr2*E_delta_smooth

				E_w_smooth = lam_smooth * \
					(self.E_w[:,smaller,ind] - self.E_w[:,larger,ind])
				
				self.E_w[:,rated_expertise_level,ind] = (1 - lr2*lam_delta) * \
					self.E_w[:,rated_expertise_level,ind]   \
					+ lr2*rated_E_w_grad - lr2*E_w_smooth

				self.E[:,:,ind] = (1 - lr2*lam_E) * self.E[:,:,ind] + lr2 * rated_E_grad

				#---- Now, the VB terms -------#

				vb_delta_smooth = lam_smooth * (self.vb_delta[:,smaller, ind] - 
							self.vb_delta[:,larger,ind])

				self.vb_delta[:,rated_expertise_level,ind] = \
					(1-lr2*lam_vf)*self.vb_delta[:,rated_expertise_level,ind] + \
					lr2*rated_vb_delta_grad - lr2*vb_delta_smooth

				vb_w_smooth = lam_smooth * (self.vb_w[:,smaller,ind] - 
							self.vb_w[:,larger,ind])

				self.vb_w[:,rated_expertise_level,ind] = \
					(1-lr2*lam_delta) * self.vb_w[:,rated_expertise_level,ind] + \
					lr2*rated_vb_w_grad -lr2*vb_w_smooth

				self.visual_bias[:,ind] = \
					(1-lr2*lam_vf) * self.visual_bias[:,ind] + lr2 * rated_vb_grad
			
			self.dd_bias = (1- lr*lam_dd) * self.dd_bias + lr*rated_dd_grad

			if unrated_bucket == 0:
				self.E2 = (1-lr2*lam_E)*self.E2 + lr2 * unrated_E_grad
				self.visual_bias2 = (1-lr2*lam_vf)*self.visual_bias2 + \
					lr2 * unrated_vb_grad
			elif unrated_bucket == 1:
				self.E3 = (1-lr2*lam_E)*self.E3 + lr2 * unrated_E_grad
				self.visual_bias3 = (1-lr2*lam_vf)*self.visual_bias3 + \
					lr2 * unrated_vb_grad

			else:				
				if unrated_bucket == 2: 
					ind = 0
				else: ind = 1 


				if unrated_expertise_level == 0:
					smaller_ = unrated_expertise_level; larger_ = unrated_expertise_level + 1
				elif unrated_expertise_level == self.nExpertise-1:
					smaller_ = unrated_expertise_level-1; larger_ = unrated_expertise_level
				else:
					smaller_ = unrated_expertise_level-1; larger_ = unrated_expertise_level+1

				'''
				Update the "E" terms.
				This includes smoothing for the deltas.
				'''

				E_delta_smooth_ = lam_smooth * \
					(self.E_delta[:,:,smaller_, ind] - self.E_delta[:,:,larger_,ind])
				

				self.E_delta[:,:,unrated_expertise_level, ind] = \
					(1-lr2*lam_E) * self.E_delta[:,:,unrated_expertise_level,ind] \
					+ lr2*unrated_E_delta_grad - lr2*E_delta_smooth_

				self.E[:,:,ind] = (1 - lr2*lam_E) * self.E[:,:,ind] + lr2*unrated_E_grad

				E_w_smooth_ = lam_smooth * \
					(self.E_w[:,smaller_,ind] - self.E_w[:,larger_,ind])

				self.E_w[:,unrated_expertise_level,ind] = (1 - lr2*lam_delta) * \
					self.E_w[:,unrated_expertise_level,ind] \
					+ lr2*unrated_E_w_grad - lr2*E_w_smooth_

				'''
				Update the visual bias terms.
				This includes smoothing for the deltas.
				'''


				vb_delta_smooth_ = lam_smooth * (self.vb_delta[:,smaller_,ind] - 
							self.vb_delta[:,larger_,ind])

				self.vb_delta[:,unrated_expertise_level,ind] = \
					(1-lr2*lam_vf)*self.vb_delta[:,unrated_expertise_level,ind] + \
					lr2*unrated_vb_delta_grad - lr2*vb_delta_smooth_

				self.visual_bias[:,ind] = (1-lr2*lam_vf) * self.visual_bias[:,ind] + lr2 * unrated_vb_grad
				
				vb_w_smooth_ = lam_smooth * (self.vb_w[:,smaller_,ind] - 
							self.vb_w[:,larger_,ind])

				self.vb_w[:,unrated_expertise_level,ind] = \
					(1-lr2*lam_delta) * self.vb_w[:,unrated_expertise_level,ind] + \
					lr2*unrated_vb_w_grad - lr2*vb_w_smooth_

				'''Finally, update DD bias'''		
			self.dd_bias = (1- lr*lam_dd) * self.dd_bias + lr*unrated_dd_grad
			
			self.visual_users[user] = (1-lr2*lam_vu) * visual_user + lr2*vu_grad

			count += 1

			if count % validation_freq is 0:
				self.validation_aucs.append(
					self.AUC(valid)
				)
		if len(self.validation_aucs):
			print ('Best accuracy: {}'.format(max(self.validation_aucs)))



	def get_bucket_objective(self, full_samples, E, vb, batch_size=200):
		item_to_artist = self.item_to_artist;
		visual_users = self.visual_users
		visual_data = self.visual_data

		objectives = {}

		n_samples = len(full_samples)
		start = 0; end = start + batch_size
		pbar = tqdm(total = len(full_samples)/batch_size)

		# Define Tensorflow graph
		tf.reset_default_graph()
		vis_data = tf.placeholder(tf.float32, shape=(None, 4096))
		user = tf.placeholder(tf.float32, shape=(None, self.n))

		# New part of graph in BAVBPR: choosing between buckets.
		E_bucket_diff_tensor = tf.convert_to_tensor(E,dtype=tf.float32)
		vb_bucket_diff_tensor = tf.convert_to_tensor(vb,dtype=tf.float32)

		encoded_b = tf.einsum('ki,lk->il', E_bucket_diff_tensor, vis_data)
		visual_interaction_b = tf.einsum('ij,ji->i', user, encoded_b)
		vis_bias_b = tf.einsum('ij,j->i', vis_data, vb_bucket_diff_tensor)
		obj_b = visual_interaction_b + vis_bias_b

		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)


		while start < n_samples:
			end = min(end, n_samples)
			samples = full_samples[start:end]

			rated_artist = [item_to_artist[x] for x in samples[:,1]]
			unrated_artist = [item_to_artist[x] for x in samples[:,2]]
			visual_user = np.array([visual_users[x] for x in samples[:,0]])
			rated_visual_data = np.array([visual_data[item] for item in samples[:,1]])
			unrated_visual_data = np.array([visual_data[item] for item in samples[:,2]])

			rated_obj = sess.run(obj_b, feed_dict={
							vis_data: rated_visual_data,
							user: visual_user,
							})

			unrated_obj = sess.run(obj_b, feed_dict={
							vis_data: unrated_visual_data,
							user: visual_user,
							})



			for i, (rated, unrated) in enumerate(zip(rated_artist, unrated_artist)):
				rated_item = samples[i,1]; unrated_item = samples[i,2]

				if rated not in objectives:
					objectives[rated] = rated_obj[i]
				else:
					objectives[rated] += rated_obj[i]

				if unrated not in objectives:
					objectives[unrated] = -unrated_obj[i]
				else:
					objectives[unrated] -= unrated_obj[i]

			start += batch_size; end += batch_size
			pbar.update(1)
		pbar.close()
		sess.close()
		return objectives




	def get_assignment_objective(self, full_samples, vb, vb_delta, vb_w, E, E_delta, E_w, batch_size=200):
		item_to_artist = self.item_to_artist;
		visual_users = self.visual_users
		visual_data = self.visual_data

		objectives = {}
		    
		n_samples = len(full_samples)
		start = 0; end = start + batch_size
		pbar = tqdm(total = len(full_samples)/batch_size)

		# Define Tensorflow graph
		tf.reset_default_graph()
		vis_data = tf.placeholder(tf.float32, shape=(None, 4096))
		user = tf.placeholder(tf.float32, shape=(None, self.n))

		visual_bias_tensor = tf.convert_to_tensor(vb, dtype=tf.float32)
		vb_delta_tensor = tf.convert_to_tensor(vb_delta, dtype=tf.float32)
		vb_w_tensor = tf.convert_to_tensor(vb_w, dtype=tf.float32)
		E_delta_tensor = tf.convert_to_tensor(E_delta, dtype=tf.float32)
		E_tensor = tf.convert_to_tensor(E, dtype=tf.float32)
		E_w_tensor = tf.convert_to_tensor(E_w, dtype=tf.float32)

		encoded = tf.einsum('ki,lk->il', E_tensor, vis_data)
		encoded = tf.einsum('ij,ik->jik', E_w_tensor, encoded)
		encoded_delta = tf.einsum('kji,lk->ijl', E_delta_tensor, vis_data)
		full_encoded = encoded+encoded_delta
		visual_interaction = tf.einsum('ij,kji->ik', user, full_encoded)

		full_visual_bias = vb_delta_tensor + tf.einsum('ij,i->ij', vb_w_tensor, visual_bias_tensor)
		vis_bias = tf.matmul(vis_data, full_visual_bias)
		obj = visual_interaction + vis_bias

		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

		old_item = -1 

		while start < n_samples:
			end = min(end, n_samples)
			samples = full_samples[start:end]

			rated_artist = [item_to_artist[x] for x in samples[:,1]]
			unrated_artist = [item_to_artist[x] for x in samples[:,2]]
			visual_user = np.array([visual_users[x] for x in samples[:,0]])
			rated_visual_data = np.array([visual_data[item] for item in samples[:,1]])
			unrated_visual_data = np.array([visual_data[item] for item in samples[:,2]])

			rated_obj = sess.run(obj, feed_dict={
							vis_data: rated_visual_data,
							user: visual_user,
							})

			unrated_obj = sess.run(obj, feed_dict={
							vis_data: unrated_visual_data,
							user: visual_user,
							})



			for i, (rated, unrated) in enumerate(zip(rated_artist, unrated_artist)):
				rated_item = samples[i,1]; unrated_item = samples[i,2]

				if rated not in objectives:
					objectives[rated] = {rated_item: rated_obj[i]}
				else:
					seen_art = objectives[rated]
					if rated_item not in seen_art:
						objectives[rated][rated_item] = rated_obj[i]
					else:
						objectives[rated][rated_item] += rated_obj[i]

				if unrated not in objectives:
					objectives[unrated] = {unrated_item: -unrated_obj[i]}
				else:
					seen_art = objectives[unrated]
					if unrated_item not in seen_art:
						objectives[unrated][unrated_item] = -unrated_obj[i]
					else:
						objectives[unrated][unrated_item] += -unrated_obj[i] #TODO: minus??

			start += batch_size; end += batch_size
			pbar.update(1)
		pbar.close()
		sess.close()
		return objectives


	def assign_classes(self):

		def DP_subproblem(nExpertise, artist_dict, chrono_art):
			buffer = np.zeros((len(chrono_art), nExpertise))
			previous_pointers = np.zeros((len(chrono_art), nExpertise))

			for i, art_number in enumerate(chrono_art):
				if art_number in artist_dict:
					objs = artist_dict[art_number] 
				else:
					objs = np.zeros(nExpertise)
				for expert_level in range (0, nExpertise):
					if i == 0:
						buffer[i, expert_level] = objs[expert_level]
						previous_pointers[i, expert_level] = -1 #base case
					else:
						if expert_level == 0:
							buffer[i, expert_level] = buffer[i-1, expert_level] \
							+ objs[expert_level]
							previous_pointers[i, expert_level] = 0 #obviously, can't go down in skill
						else:
							same_level = buffer[i-1, expert_level]
							previous_level = buffer[i-1, expert_level-1] #i.e. this one the artist "leveled up"
							if same_level > previous_level: #Pick the one iwth the higher obj, i.e. same_level
								buffer[i, expert_level] = same_level + objs[expert_level]
								previous_pointers[i, expert_level] = expert_level
							else:
								buffer[i, expert_level] = previous_level + objs[expert_level]
								previous_pointers[i, expert_level] = expert_level-1


			best_subsequence = np.zeros((len(chrono_art)), dtype=int)
			current_expert_level = np.argmax(buffer[-1,:])
			obj_max = np.max(buffer[-1,:])

			curr = -1
			best_subsequence[-1] = current_expert_level
			while previous_pointers[curr, expert_level] != -1: #-1 indicates "end"
				current_expert_level = previous_pointers[curr, int(current_expert_level)]
				curr -= 1
				best_subsequence[curr] = current_expert_level
			return best_subsequence, obj_max


		bucket_obj1 = self.get_bucket_objective(self.assign_triples, self.E2, self.visual_bias2)
		bucket_obj2 = self.get_bucket_objective(self.assign_triples, self.E3, self.visual_bias3)
		obj = self.get_assignment_objective(assign_triples, 
			self.visual_bias[:,0], self.vb_delta[:,:,0], self.vb_w[:,:,0], 
			self.E[:,:,0], self.E_delta[:,:,:,0], self.E_w[:,:,0])
		obj2 = self.get_assignment_objective(assign_triples, 
			self.visual_bias[:,1], self.vb_delta[:,:,1], self.vb_w[:,:,1], 
			self.E[:,:,1], self.E_delta[:,:,:,1], self.E_w[:,:,1])

		for artist in obj:
			artworks = self.artist_dict[artist]
			best_subsequence, obj_max = DP_subproblem(self.nExpertise, obj[artist], artworks)
			best_subsequence2, obj_max2 = DP_subproblem(self.nExpertise, obj2[artist], artworks)

			if obj_max > bucket_obj1[artist] and obj_max > bucket_obj2[artist] and obj_max > obj_max2:
				self.artwork_assignments[artist] = dict(zip(artworks, best_subsequence))
				self.artist_assignments[artist] = 2
			elif obj_max2 > bucket_obj1[artist] and obj_max2 > bucket_obj2[artist]:
				self.artwork_assignments[artist] = dict(zip(artworks, best_subsequence2))
				self.artist_assignments[artist] = 3
			else: 
				if bucket_obj1[artist] > bucket_obj2[artist]:
					self.artist_assignments[artist] = 0
				else:
					self.artist_assignments[artist] = 1



	#----- Plots and other outputs --------#

	def assignment_histogram(self, bucket):
		mylist = []
		for artist in self.artist_assignments:
			if self.artist_assignments[artist] == bucket:
				mylist.append(list(self.artwork_assignments[artist].values()))

		mylist = [item for sublist in mylist for item in sublist]
		plt.hist(mylist)
		plt.show()



	def plot_user_expertise_progression(self, bucket):
		i = 0 
		fig, axes = plt.subplots(nrows=5, ncols=2, figsize = (15,15))
		keys = list(self.artist_assignments.keys())
		np.random.shuffle(keys)
		for artist in keys:
			if self.artist_assignments[artist] != bucket: continue
			curr_assignments = self.artwork_assignments[artist]
			vals = list(curr_assignments.values())
			x = self.artist_dict[artist]
			ratings = [(self.img_nfavs_dict[img]) for img in x]
			y = [np.log(self.img_nfavs_dict[img] + 1) for img in x]
			axes[i%5, int(i/5)].scatter(range(len(y)), y, c=vals)
			i += 1
			if i == 10: break
		plt.show()


	def plot_user_bucket(self,bucket_number):
		i = 0 
		fig, axes = plt.subplots(nrows=5, ncols=2)
		keys = list(self.artist_assignments.keys())
		np.random.shuffle(keys)
		for artist in keys:
			curr_assignments = self.artist_assignments[artist]
			if isinstance(curr_assignments, dict): continue
			if curr_assignments != bucket_number: continue
			x = self.artist_dict[artist]
			ratings = [(self.img_nfavs_dict[img]) for img in x]
			y = [np.log(self.img_nfavs_dict[img] + 1) for img in x]
			axes[i%5, int(i/5)].scatter(range(len(y)), y)
			i += 1
			if i == 10: break
		plt.show()


	def print_average_nfavs_per_level(self, bucket):
		mylist = []
		artist_assignments = self.artist_assignments
		for artist in artist_assignments:
			if artist_assignments[artist] != bucket: continue
			mylist.append(list(zip(list(self.artwork_assignments[artist].keys()), 
					list(self.artwork_assignments[artist].values()))))

		mylist = [item for sublist in mylist for item in sublist]

		img_nfavs_dict = self.img_nfavs_dict
		mylist = [(y, img_nfavs_dict[x]) for (x,y) in mylist]
		grouped = []
		for i in range(self.nExpertise):
			grouped.append([x for x in mylist if x[0] == i])

		for i in range(self.nExpertise):
			curr = grouped[i]
			try:
				avg = np.median([x[1] for x in curr])
				avg2 = np.mean([x[1] for x in curr])
			except:
				avg = 0
			print("Average nFavs for level {} is {}".format(i, [avg, avg2]))


	def plot_bucket_counts(self):
		bucket_count = [0, 0, 0, 0];
		for artist in self.artist_assignments:
			bucket = self.artist_assignments[artist]
			bucket_count[bucket] += 1
		names = ['Bucket 0:  Beginner', 'Bucket 1: Expert', 
			'Bucket 2: Progression', 'Bucket 3: Progression']
		plt.bar([0, 1, 2, 3], bucket_count, tick_label = names)
		print (bucket_count)
		plt.show()

	def average_per_bucket(self):
		buckets = [[], [], [], []]
		for artist in self.artist_assignments:
			artworks = self.artist_dict[artist]
			nFavs = [self.img_nfavs_dict[x] for x in artworks]
			bucket_num = self.artist_assignments[artist]
			buckets[bucket_num].append(np.mean(nFavs))
		print('Average nFavs for Bucket 0 is {}, Bucket 1 is {}, Bucket 2 is {}, Bucket 3 is {}'.format(
				np.mean(buckets[0]), np.mean(buckets[1]), np.mean(buckets[2]), np.mean(buckets[3])))


if __name__ == '__main__':

	from data import Data 
	data = Data(False, False)
	fn = '../cache/VBPR_3_3_0.5_0.007_default_reg.pkl'
	bavbpr = BAVBPR(*data.get_max(), 
		filename=fn, lr=0.0005, lr2=0.000005, nExpertise=3, 
		lam_vf = 1, lam_E = 1)
	valid_data = data.generate_evaluation_samples(True)
	assign_triples = data.generate_assignment_triples(1)
	bavbpr.set_visual_data(data.get_visual_data())
	bavbpr.set_dd_dict(data.get_dd_dict())
	bavbpr.set_assignment_triples(assign_triples)
	bavbpr.set_dicts(*data.get_artist_dicts())
	bavbpr.initialize_assignments()
	for i in range(3):
		train_data = data.generate_train_samples(1000000)
		bavbpr.train(train_data, valid_data, validation_freq=1000000)
		bavbpr.assign_classes()
		bavbpr.validation_aucs.append(bavbpr.AUC(valid_data))
		import pdb; pdb.set_trace()
	bavbpr.plot_validation_error()

