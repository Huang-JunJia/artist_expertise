import numpy as np
from tqdm import tqdm
from copy import deepcopy
import math
import warnings
import itertools 

import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pkl
from vbpr import VBPR

import numpy as np
from collections import Counter
np.seterr(over='warn', invalid='warn')
warnings.filterwarnings('error')

def sigmoid(x):
	if x < -709:
		sigmoid = 0.0
	else:
		sigmoid = 1.0 / (1.0 + math.exp(-x))
	return sigmoid

class AVBPR(VBPR):

	def __init__(self, max_item, max_user, filename, k=3, lr=0.0005, lam_u=0.1, 
				lam_bias=0.1, lam_rated=0.1, lam_unrated=0.1,
				lam_vf = .1, lam_E = .10, lam_vu = .10, lam_dd=0.1,
				n=3, lr2=0.000005, nExpertise=5, lam_smooth=0.1, lam_delta=0.1):

		self.filename = filename 

		self.max_item = max_item
		self.max_user = max_user

		self.validation_aucs = []
		self.set_hyperparameters(k, n, lr, lr2, lam_u, lam_bias, lam_rated, 
								lam_unrated, lam_vf, lam_E, lam_vu, lam_dd, 
								lam_smooth, lam_delta, nExpertise)

		self.__initialize_parameters(filename)
		self.assignments = []
		self.average_nfavs = []

	def set_hyperparameters(self, k, n, lr, lr2, lam_u, lam_bias, 
			lam_rated, lam_unrated, lam_vf, lam_E, lam_vu, lam_dd, 
			lam_smooth, lam_delta, nExpertise):

		self.k = k; self.lr = lr; self.lam_u = lam_u; self.lam_bias = lam_bias
		self.lam_rated = lam_rated; self.lam_unrated = lam_unrated
		self.lam_vf = lam_vf; self.lam_E = lam_E; self.lam_vu = lam_vu
		self.n = n; self.lr2 = lr2; self.lam_smooth = lam_smooth; 
		self.nExpertise = nExpertise; self.lam_dd = lam_dd; self.lam_delta = lam_delta

	def set_artist_data(self, artist_dict, item_to_artist):
		self.item_to_artist = item_to_artist
		self.artist_dict = artist_dict

	def set_assignment_triples(self, assign_triples):
		self.assign_triples = assign_triples

	def set_dicts(self, artist_dict, item_to_artist, img_nfavs_dict):
		self.artist_dict, self.item_to_artist, self.img_nfavs_dict = \
			artist_dict, item_to_artist, img_nfavs_dict

	def set_data_dicts(self, train, valid, test, removed):
		self.train_dict = train
		self.valid_dict= valid
		self.test_dict = test
		self.removed_items = removed

	#-----------------------------#

	def __initialize_parameters(self, filename):
		
		with open(filename, 'rb') as f:
			self.item_biases, self.latent_items, self.latent_users, \
				self.visual_users, self.E, self.visual_bias, self.dd_bias = \
				pkl.load(f) 

		'''Now for the expertise parameters'''
		self.dd_bias = [self.dd_bias]*self.nExpertise

		self.E_delta = np.zeros(shape=(4096, self.n, self.nExpertise))
		self.vb_delta = np.zeros(shape=(4096, self.nExpertise))

		'''Per-factor scaling'''
		self.E_w = np.ones(shape=(self.n, self.nExpertise))*1E-50
		self.vb_w = np.ones(shape=(4096, self.nExpertise))*1E-50

	def initialize_assignments(self):
		artist_assignments = {}
		artwork_assignments = {}
		for key in self.artist_dict:
			artworks = self.artist_dict[key]
			div = math.ceil(float(len(artworks))/self.nExpertise) 
			expertise = [int(i/div) for i, x in enumerate(artworks)]
			artist_assignments[key] = dict(zip(artworks, expertise))

			for art, level in zip(artworks, expertise):
				artwork_assignments[art] = level
		self.artist_assignments = artist_assignments	
		self.artwork_assignments = artwork_assignments



	def initialize_assignments2(self):
		artist_assignments = {}
		artwork_assignments = {}

		for key in self.artist_dict:
			artworks = self.artist_dict[key]
			nFavs = [self.img_nfavs_dict[x] for x in artworks]
			expertise = []; curr = 0
			for artwork in artworks:
				#if curr < 1 and self.img_nfavs_dict[artwork] > 110:
				curr=0 #delete me too
				if self.img_nfavs_dict[artwork] > 25:
					curr = 1
				#if curr < 2 and self.img_nfavs_dict[artwork] > 300:
				if self.img_nfavs_dict[artwork] > 300:
					curr = 2

				'''if curr < 3 and self.img_nfavs_dict[artwork] > 600:
					curr = 3

				if curr < 4 and self.img_nfavs_dict[artwork] > 1200:
					curr = 4'''
				artwork_assignments[artwork] = curr
				expertise.append(curr)
			artist_assignments[key] = dict(zip(artworks, expertise))

		self.artist_assignments = artist_assignments
		self.artwork_assignments = artwork_assignments


	def generate_data(self, nSamples=1000000):
		#-------- Create local variables for convenience and brevity ------#
		train = self.train_dict
		valid = self.valid_dict
		test = self.test_dict

		removed_items = self.removed_items
		max_item = self.max_item

		artwork_assignments = self.artwork_assignments
		#------------------------------------------------------------------#

		samples = np.zeros((nSamples, 3), dtype=int)
		keys = list(train.keys())
		users = np.random.choice(keys, nSamples) 
		samples[:, 0] = users

		for i, user in enumerate(users):
			rated_items = train[user]
			valid_item = valid[user]

			if test: test_item = test[user]
			else: test_item = None

			rated_item = np.random.choice(rated_items)
			rated_bucket = artwork_assignments[rated_item]
			unrated_item = np.random.choice(max_item)
			if unrated_item in artwork_assignments:
				unrated_bucket = artwork_assignments[unrated_item]
			else:
				unrated_bucket = -1

			while unrated_item in rated_items or unrated_item in removed_items \
				or unrated_item == valid_item or unrated_item == test_item or \
				rated_bucket != unrated_bucket:

				unrated_item = np.random.choice(max_item)
				if unrated_item in artwork_assignments:
					unrated_bucket = artwork_assignments[unrated_item]
				else:
					unrated_bucket = -1
			samples[i, 1] = rated_item
			samples[i, 2] = unrated_item

		return samples
	

	#================ Implementation details ================#

	def BR_opt(self, rated_item_bias, unrated_item_bias, latent_user, 
		latent_rated_item, latent_unrated_item, rated_vf, unrated_vf, 
		encoded_difference, rated_vb, unrated_vb, visual_user, dd_difference):

		bias_difference = rated_item_bias - unrated_item_bias
		latent_difference = np.dot(latent_rated_item - latent_unrated_item, 
				latent_user)
		visual_difference = np.dot(encoded_difference, visual_user)

		rated_visual_bias = np.dot(rated_vf, rated_vb)
		unrated_visual_bias = np.dot(unrated_vf, unrated_vb)
		visual_bias_difference = rated_visual_bias - unrated_visual_bias
		
		return (bias_difference + latent_difference + visual_difference \
			+ visual_bias_difference + dd_difference)
	

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

			#Get the artists' expertise level for that artwork.
			rated_expertise_level = \
				self.artist_assignments[rated_artist][rated_item]
			unrated_expertise_level = \
				self.artist_assignments[unrated_artist][unrated_item]

			'''The encoding matrix part has three parts:
				1) The baseline model E
				2) The per-factor weights E_w 
				3) The change in matrix weights E_delta
			'''
			rated_E_delta = self.E_delta[:,:,rated_expertise_level]
			rated_E_w = self.E_w[:,rated_expertise_level]
			rated_encoded = np.dot(np.transpose(self.E), rated_vf)
			rated_delta_encoded = np.dot(np.transpose(rated_E_delta), rated_vf)
			rated_encoded = np.multiply(rated_E_w, rated_encoded)+\
							rated_delta_encoded

			unrated_E_delta = self.E_delta[:,:,unrated_expertise_level]
			unrated_E_w = self.E_w[:,unrated_expertise_level]
			unrated_encoded = np.dot(np.transpose(self.E), unrated_vf)
			unrated_delta_encoded = np.dot(np.transpose(unrated_E_delta), unrated_vf)
			unrated_encoded = np.multiply(unrated_E_w, unrated_encoded)+\
								unrated_delta_encoded

			encoded_difference = rated_encoded-unrated_encoded
			'''The visual bias is similar.
			'''
			rated_vb_delta = self.vb_delta[:,rated_expertise_level]
			rated_vb_w = self.vb_w[:, rated_expertise_level]
			rated_visual_bias = rated_vb_delta +\
				np.multiply(self.visual_bias, rated_vb_w)

			unrated_vb_delta = self.vb_delta[:,unrated_expertise_level]
			unrated_vb_w = self.vb_w[:, unrated_expertise_level]
			unrated_visual_bias = unrated_vb_delta +\
				np.multiply(self.visual_bias, unrated_vb_w)

								
			dd_difference = \
				self.dd_dict[rated_item]*self.dd_bias[rated_expertise_level]-\
                self.dd_dict[unrated_item]*self.dd_bias[unrated_expertise_level]

			# Determine the output, i.e. x_uij 
			br_opt = self.BR_opt(rated_item_bias, unrated_item_bias, latent_user, 
				latent_rated_item, latent_unrated_item, rated_vf, unrated_vf,
				encoded_difference, rated_visual_bias, unrated_visual_bias, 
				visual_user, dd_difference)

			auc += (br_opt > 0)

		return auc/len(samples)


	def train(self, samples, valid, validation_freq=250000):

		count = 1

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

			rated_expertise_level = self.artist_assignments[rated_artist][rated_item]
			unrated_expertise_level = self.artist_assignments[unrated_artist][unrated_item]

			'''The encoding matrix part has three parts:
				1) The baseline model E
				2) The per-factor weights E_w 
				3) The change in matrix weights E_delta
			'''
			rated_E_delta = self.E_delta[:,:,rated_expertise_level]
			rated_E_w = self.E_w[:,rated_expertise_level]
			rated_encoded = np.dot(np.transpose(self.E), rated_vf)
			rated_delta_encoded = np.dot(np.transpose(rated_E_delta), rated_vf)
			
			rated_encoded_sum = np.multiply(rated_E_w, rated_encoded)+\
						rated_delta_encoded

			unrated_E_delta = self.E_delta[:,:,unrated_expertise_level]
			unrated_E_w = self.E_w[:,unrated_expertise_level]
			unrated_encoded = np.dot(np.transpose(self.E), unrated_vf)
			unrated_delta_encoded = np.dot(np.transpose(unrated_E_delta), unrated_vf)
			unrated_encoded_sum = np.multiply(unrated_E_w, unrated_encoded)+\
								unrated_delta_encoded

			encoded_difference = rated_encoded_sum-unrated_encoded_sum
			'''The visual bias is similar.
			'''
			rated_vb_delta = self.vb_delta[:,rated_expertise_level]
			rated_vb_w = self.vb_w[:, rated_expertise_level]
			rated_visual_bias = rated_vb_delta +\
				np.multiply(self.visual_bias, rated_vb_w)

			unrated_vb_delta = self.vb_delta[:,unrated_expertise_level]
			unrated_vb_w = self.vb_w[:, unrated_expertise_level]
			unrated_visual_bias = unrated_vb_delta +\
				np.multiply(self.visual_bias, unrated_vb_w)


			rated_dd = self.dd_dict[rated_item]
			unrated_dd = self.dd_dict[unrated_item]

			rated_dd_bias = self.dd_bias[rated_expertise_level]
			unrated_dd_bias = self.dd_bias[unrated_expertise_level]

			dd_difference = \
				rated_dd*rated_dd_bias - unrated_dd*unrated_dd_bias


			br_opt = self.BR_opt(rated_item_bias, unrated_item_bias, latent_user, 
				latent_rated_item, latent_unrated_item, rated_vf, unrated_vf,
				encoded_difference, rated_visual_bias, unrated_visual_bias, 
				visual_user, dd_difference)

			output = sigmoid(-br_opt)



			# ====== Get Gradients ===========================
			ib_grad = output
			li_grad = output * latent_user
			lu_grad = output * (latent_rated_item-latent_unrated_item)

			'''E_delta is ok'''
			rated_E_delta_grad = output * (
					np.dot(rated_vf[:,None],visual_user[None,:]))
			unrated_E_delta_grad = -output * (
				np.dot(unrated_vf[:,None],visual_user[None,:]))

			rated_E_w_grad = output * np.multiply(rated_encoded, visual_user)
			unrated_E_w_grad = -output * np.multiply(unrated_encoded, visual_user)

			E_grad = np.outer(rated_vf, np.multiply(visual_user, rated_E_w)) 
			E_grad -= np.outer(unrated_vf, np.multiply(visual_user, unrated_E_w))
			E_grad = output * E_grad 


			vu_grad = output * (encoded_difference)

			rated_vb_delta_grad = output * rated_vf
			unrated_vb_delta_grad = -output * unrated_vf

			rated_vb_w_grad = output * np.multiply(rated_vf, self.visual_bias)
			unrated_vb_w_grad = -output * np.multiply(unrated_vf, self.visual_bias)

			vb_grad = output * (np.multiply(rated_vb_w, rated_vf) -
								np.multiply(unrated_vb_w, unrated_vf))


			rated_dd_grad = output * rated_dd 
			unrated_dd_grad = -output * unrated_dd


			# ================================================

			# Perform gradient updates
			self.item_biases[rated_item] = (1-lr*lam_bias) * rated_item_bias + lr*ib_grad
			self.item_biases[unrated_item] = (1-lr*lam_bias) * unrated_item_bias - lr*ib_grad

			self.latent_items[rated_item] = (1-lr*lam_rated) * latent_rated_item + lr*li_grad 
			self.latent_items[unrated_item] = (1-lr*lam_unrated) * latent_unrated_item - lr*li_grad 

			self.latent_users[user] = (1-lr*lam_unrated) * latent_user + lr*lu_grad

			'''
			Get the smoothing expertise levels
			'''
			if rated_expertise_level == 0:
				smaller = rated_expertise_level; larger = rated_expertise_level + 1
			elif rated_expertise_level == self.nExpertise-1:
				smaller = rated_expertise_level-1; larger = rated_expertise_level
			else:
				smaller = rated_expertise_level-1; larger = rated_expertise_level+1

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

			E_delta_smooth = lam_smooth * \
				(self.E_delta[:,:,smaller] - self.E_delta[:,:,larger])
			E_delta_smooth_ = lam_smooth * \
				(self.E_delta[:,:,smaller_] - self.E_delta[:,:,larger_])

			self.E_delta[:,:,rated_expertise_level] = \
				(1-lr2*lam_E) * rated_E_delta \
				+ lr2*rated_E_delta_grad - lr2*E_delta_smooth


			self.E_delta[:,:,unrated_expertise_level] = \
				(1-lr2*lam_E) * unrated_E_delta \
				+ lr2*unrated_E_delta_grad - lr2*E_delta_smooth_

			self.E = (1 - lr2*lam_E) * self.E + lr2*E_grad

			E_w_smooth = lam_smooth * \
				(self.E_w[:,smaller] - self.E_w[:,larger])
			E_w_smooth_ = lam_smooth * \
				(self.E_w[:,smaller_] - self.E_w[:,larger_])

			self.E_w[:,rated_expertise_level] = (1 - lr2*lam_delta) * rated_E_w  \
				+ lr2*rated_E_w_grad #- lr2*E_w_smooth
			self.E_w[:,unrated_expertise_level] = (1 - lr2*lam_delta) * unrated_E_w \
				+ lr2*unrated_E_w_grad #- lr2*E_w_smooth_

			'''
			Update the visual bias terms.
			This includes smoothing for the deltas.
			'''

			self.visual_users[user] = (1-lr2*lam_vu) * visual_user + lr2*vu_grad

			vb_delta_smooth = lam_smooth * (self.vb_delta[:,smaller] - 
						self.vb_delta[:,larger])
			vb_delta_smooth_ = lam_smooth * (self.vb_delta[:,smaller_] - 
						self.vb_delta[:,larger_])

			self.vb_delta[:,rated_expertise_level] = \
				(1-lr2*lam_vf)*rated_vb_delta + \
				lr2*rated_vb_delta_grad - lr2*vb_delta_smooth
			self.vb_delta[:,unrated_expertise_level] = \
				(1-lr2*lam_vf)*unrated_vb_delta + \
				lr2*unrated_vb_delta_grad - lr2*vb_delta_smooth_

			self.visual_bias = (1-lr2*lam_vf) * self.visual_bias + lr2 * vb_grad
			
			vb_w_smooth = lam_smooth * (self.vb_w[:,smaller] - 
						self.vb_w[:,larger])
			vb_w_smooth_ = lam_smooth * (self.vb_w[:,smaller_] - 
						self.vb_w[:,larger_])

			self.vb_w[:,rated_expertise_level] = \
				(1-lr2*lam_delta) * rated_vb_w + \
				lr2*rated_vb_w_grad #-lr2*vb_w_smooth

			self.vb_w[:,unrated_expertise_level] = \
				(1-lr2*lam_delta) * unrated_vb_w + \
				lr2*unrated_vb_w_grad #- lr2*vb_w_smooth_

			'''Finally, update DD bias'''		
			self.dd_bias[rated_expertise_level] = \
				(1- lr*lam_dd) * rated_dd_bias + lr*rated_dd_grad
			self.dd_bias[unrated_expertise_level] = \
				(1- lr*lam_dd) * unrated_dd_bias + lr*unrated_dd_grad


			count += 1

			if count % validation_freq is 0:
				self.validation_aucs.append(
					self.AUC(valid)
				)

		if len(self.validation_aucs):
			print ('Best accuracy: {}'.format(max(self.validation_aucs)))


	def get_assignment_objective(self, full_samples, batch_size=200):
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

		visual_bias_tensor = tf.convert_to_tensor(self.visual_bias, dtype=tf.float32)
		vb_delta_tensor = tf.convert_to_tensor(self.vb_delta, dtype=tf.float32)
		vb_w_tensor = tf.convert_to_tensor(self.vb_w, dtype=tf.float32)
		E_delta_tensor = tf.convert_to_tensor(self.E_delta, dtype=tf.float32)
		E_tensor = tf.convert_to_tensor(self.E, dtype=tf.float32)
		E_w_tensor = tf.convert_to_tensor(self.E_w, dtype=tf.float32)

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
						objectives[unrated][unrated_item] -= unrated_obj[i] #TODO: minus??

			start += batch_size; end += batch_size
			pbar.update(1)
		pbar.close()
		return objectives


	def assign_classes(self):

		def DP_subproblem(nExpertise, artist_dict, chrono_art):
			buffer = np.zeros((len(chrono_art), nExpertise))
			previous_pointers = np.zeros((len(chrono_art), nExpertise))

			for i, art_number in enumerate(chrono_art):
				if art_number not in artist_dict or art_number in self.valid_dict:
					objs = np.zeros(nExpertise)
				else:
					objs = artist_dict[art_number] 
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

			curr = -1
			best_subsequence[-1] = current_expert_level
			while previous_pointers[curr, expert_level] != -1: #-1 indicates "end"
				current_expert_level = previous_pointers[curr, int(current_expert_level)]
				curr -= 1
				best_subsequence[curr] = current_expert_level
			for art, level in zip(chrono_art, best_subsequence):
				self.artwork_assignments[art] = level
			return best_subsequence 



		obj = self.get_assignment_objective(self.assign_triples) #returns a dict of dicts of objectives for each artwork form each artist

		old_assignments = deepcopy(self.artist_assignments)
		count = 0
		different = 0.0
		for artist in obj:
			artworks = self.artist_dict[artist]
			best_subsequence = DP_subproblem(self.nExpertise, obj[artist], artworks)
			self.artist_assignments[artist] = dict(zip(artworks, best_subsequence))
			for assignment1, assignment2 in zip(old_assignments[artist].values(), 
										self.artist_assignments[artist].values()):
				count += 1
				if assignment1 != assignment2: different +=1
		print('Proportion of artwork that changed is {}'.format(different/count))
	#-----Plots and other informational outputs-----#

	def plot_validation_error(self, validation_freq=250000):
		super().plot_validation_error(validation_freq)

	def average_nfavs_per_level(self):
		mylist = []
		artist_assignments = self.artist_assignments
		for artist in artist_assignments:
			mylist.append(list(zip(list(artist_assignments[artist].keys()), 
					list(artist_assignments[artist].values()))))

		mylist = [item for sublist in mylist for item in sublist]

		img_nfavs_dict = self.img_nfavs_dict
		mylist = [(y, img_nfavs_dict[x]) for (x,y) in mylist]
		grouped = []
		for i in range(self.nExpertise):
			grouped.append([x for x in mylist if x[0] == i])
		temp = []
		for i in range(self.nExpertise):
			curr = grouped[i]
			try:
				avg = np.median([x[1] for x in curr])
				avg2 = np.mean([x[1] for x in curr])
			except:
				avg = 0
			print("Average nFavs for level {} is {}".format(i, [avg, avg2]))
			temp.append(avg2)
		self.average_nfavs.append(temp)

	def assignments_per_level(self):
		mylist = []
		for artist in self.artist_assignments:
			mylist.append(list(self.artist_assignments[artist].values()))

		mylist = [item for sublist in mylist for item in sublist]
		ctr = Counter(mylist)
		print(dict(ctr))
		self.assignments.append(dict(ctr))


	def assignment_histogram(self):
		mylist = []
		for artist in self.artist_assignments:
			mylist.append(list(self.artist_assignments[artist].values()))

		mylist = [item for sublist in mylist for item in sublist]
		plt.hist(mylist)
		plt.show()

	def plot_user_expertise_progression(self, log=False):
		i = 0 
		fig, axes = plt.subplots(nrows=10, ncols=2, figsize = (15,15))
		keys = list(self.artist_assignments.keys())
		np.random.shuffle(keys)
		for artist in keys:
			curr_assignments = self.artist_assignments[artist]
			vals = list(curr_assignments.values())
			x = self.artist_dict[artist]
			ratings = [(self.img_nfavs_dict[img]) for img in x]
			y = [np.log(self.img_nfavs_dict[img] + 1) for img in x]
			if log: ratings=y
			axes[i%10, int(i/10)].scatter(range(len(y)), ratings, c=vals)
			i += 1
			if i == 20: break
		plt.show()

	def progression_stats(self):
		possible_choices = range(self.nExpertise)
		results = {}
		for i in range(1, self.nExpertise+1):
			curr = list(itertools.combinations(possible_choices, i))
			for sublist in curr:
				results[sublist] = 0

		for artist in self.artist_assignments:
			curr_assignments = self.artist_assignments[artist]
			vals = list(curr_assignments.values())
			for key in results:
				if set(key).issubset(vals): results[key]+=1

		for key in results:
			print('For {}, count is {}'.format(key, results[key]))


	def visualize_artists_scheme(self, wanted_items, unwanted_items, log=False):
		artists = list(self.artist_assignments.keys())
		np.random.shuffle(artists)
		
		i = 0 
		fig, axes = plt.subplots(nrows=5, ncols=2, figsize = (15,15))

		for artist in artists:
			curr_assignments = self.artist_assignments[artist]
			vals = list(curr_assignments.values())
			x = self.artist_dict[artist]
			ratings = [(self.img_nfavs_dict[img]) for img in x]
			y = [np.log(self.img_nfavs_dict[img] + 1) for img in x]

			if not set(wanted_items).issubset(vals): continue
			if set(unwanted_items).intersection(vals): continue
			if log: ratings = y
			axes[i%5, int(i/5)].scatter(range(len(y)), ratings, c=vals)
			i += 1
			if i == 10: break

		plt.show()

	def powerset_mean(self, wanted_items, unwanted_items=(), mean_of_means=False):
		artists = list(self.artist_assignments.keys())
		results = []
		wanted = [[] for i in range(len(wanted_items))]

		for artist in artists:
			curr_assignments = self.artist_assignments[artist]
			vals = list(curr_assignments.values())
			x = self.artist_dict[artist]
			ratings = [(self.img_nfavs_dict[img]) for img in x]

			if not set(wanted_items).issubset(vals): continue
			if set(unwanted_items).intersection(vals): continue
			
			results.append(np.mean(ratings))
			for i, el in enumerate(wanted_items):
				temp = [rating for i, rating in enumerate(ratings) 
					if vals[i] == el]
				if mean_of_means:
					wanted[i].append(temp)
				else:
					wanted[i].append(np.mean(temp))
		
		print ('Average nFavs for this scheme is {} for {} items'.format(np.mean(results), len(results)))
		for i, level in enumerate(wanted):
			if mean_of_means:
				level = [item for sublist in level for item in sublist]
				avg = np.mean(level)
			else:
				avg = np.mean(level)
			print ('Average nfavs for {} is {}'.format(wanted_items[i], avg,))

if __name__ == '__main__':
	plt.ion()
	from data import Data 
	data = Data(False, False, 100)
	hard_data = Data(False, True, 5)

	fn = '../cache/VBPR_{}_{}_{}_{}_default_reg_100.pkl'.format(3, 3, 0.5, 0.007, 'hard')
	#fn = '../cache/VBPR_{}_{}_{}_{}_default_reg.pkl'.format(3, 3, 0.5, 0.007, 'hard')
	#fn = '../cache/random.pkl'
	avbpr = AVBPR(*data.get_max(), filename=fn, lr=0.00, lr2=1E-7,
			k=3, n=3, nExpertise=5, 
			lam_vf=.10, lam_E=.10, lam_vu=.10, lam_smooth=.10, lam_delta=.10)
	valid_data = data.generate_evaluation_samples(True)
	assign_triples = hard_data.generate_train_samples(20000000)
	avbpr.set_visual_data(data.get_visual_data())
	avbpr.set_data_dicts(*data.get_data_dicts())
	avbpr.set_dd_dict(data.get_dd_dict())
	avbpr.set_assignment_triples(assign_triples)
	avbpr.set_dicts(*data.get_artist_dicts())
	avbpr.initialize_assignments()
	avbpr.hard_valid_aucs = []
	hard_valid_data = hard_data.generate_evaluation_samples(True)
	avbpr.average_nfavs_per_level()
	avbpr.assignments_per_level()
	for i in range(5):
		train_data = avbpr.generate_data(5000000)
		avbpr.train(train_data, valid_data, validation_freq=len(train_data))
		avbpr.hard_valid_aucs.append(avbpr.AUC(hard_valid_data))
		avbpr.assign_classes()
		avbpr.validation_aucs.append(avbpr.AUC(valid_data))
		avbpr.hard_valid_aucs.append(avbpr.AUC(hard_valid_data))
		avbpr.update_lr(1, 0.5)
		avbpr.average_nfavs_per_level()
		avbpr.assignments_per_level()

	avbpr.plot_validation_error()
	import pdb; pdb.set_trace()