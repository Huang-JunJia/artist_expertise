import numpy as np
from tqdm import tqdm
import math
import warnings

import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pkl
from vbpr import VBPR


def sigmoid(x):
	if x < -709:
		sigmoid = 0.0
	else:
		sigmoid = 1.0 / (1.0 + math.exp(-x))
	return sigmoid

class AVBPR(VBPR):

	def __init__(self, max_item, max_user, filename, k=3, lr=0.0005, lam_u=0.01, 
				lam_bias=0.01, lam_rated=0.01, lam_unrated=0.01,
				lam_vf = .1, lam_E = .10, lam_vu = .10, lam_dd=0.01,
				n=3, lr2=0.000005, nExpertise=5, lam_smooth=0.1, lam_delta=0.01):

		self.filename = filename 

		self.max_item = max_item
		self.max_user = max_user

		self.validation_aucs = []
		self.set_hyperparameters(k, n, lr, lr2, lam_u, lam_bias, lam_rated, 
								lam_unrated, lam_vf, lam_E, lam_vu, lam_dd, 
								lam_smooth, lam_delta, nExpertise)

		self.__initialize_parameters(filename)

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

	def __initialize_parameters(self, filename):
		
		self.item_biases, self.latent_items, self.latent_users, \
			self.visual_users, self.E, self.visual_bias, self.dd_bias = \
			pkl.load(open(filename, 'rb')) 

		'''Now for the expertise parameters'''
		self.dd_bias = [self.dd_bias]*self.nExpertise

		self.E_delta = np.zeros(shape=(4096, self.n, self.nExpertise))
		self.vb_delta = np.zeros(shape=(4096, self.nExpertise))

		'''Per-factor scaling'''
		self.E_w = np.zeros(shape=(self.n, self.nExpertise))
		self.vb_w = np.zeros(shape=(4096, self.nExpertise))

	def initialize_assignments(self):
		artist_assignments = {}

		for key in self.artist_dict:
			artworks = self.artist_dict[key]
			div = math.ceil(float(len(artworks))/self.nExpertise) 
			expertise = [int(i/div) for i, x in enumerate(artworks)]
			artist_assignments[key] = dict(zip(artworks, expertise))
		self.artist_assignments = artist_assignments	

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
			try:
				rated_encoded_sum = np.multiply(rated_E_w, rated_encoded)+\
							rated_delta_encoded
			except RuntimeWarning: import pdb; pdb.set_trace()

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
				+ lr2*rated_E_w_grad - lr2*E_w_smooth
			self.E_w[:,unrated_expertise_level] = (1 - lr2*lam_delta) * unrated_E_w \
				+ lr2*unrated_E_w_grad - lr2*E_w_smooth_

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
				lr2*rated_vb_w_grad -lr2*vb_w_smooth

			self.vb_w[:,unrated_expertise_level] = \
				(1-lr2*lam_delta) * unrated_vb_w + \
				lr2*unrated_vb_w_grad - lr2*vb_w_smooth_

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

		visual_bias_tensor = tf.convert_to_tensor(self.vb_delta, dtype=tf.float32)
		E_tensor = tf.convert_to_tensor(self.E_delta, dtype=tf.float32)


		encoded = tf.einsum('kji,lk->ijl', E_tensor, vis_data)
		visual_interaction = tf.einsum('ij,kji->ik', user, encoded)

		vis_bias = tf.matmul(vis_data, visual_bias_tensor)
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

			curr = -1
			best_subsequence[-1] = current_expert_level
			while previous_pointers[curr, expert_level] != -1: #-1 indicates "end"
				current_expert_level = previous_pointers[curr, int(current_expert_level)]
				curr -= 1
				best_subsequence[curr] = current_expert_level
			return best_subsequence 



		obj = self.get_assignment_objective(self.assign_triples) #returns a dict of dicts of objectives for each artwork form each artist


		for artist in obj:
			artworks = self.artist_dict[artist]
			best_subsequence = DP_subproblem(self.nExpertise, obj[artist], artworks)
			self.artist_assignments[artist] = dict(zip(artworks, best_subsequence))
	
	#-----Plots and other informational outputs-----#

	def plot_validation_error(self, validation_freq=250000):
		super().plot_validation_error(validation_freq)

	def print_average_nfavs_per_level(self):
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

		for i in range(self.nExpertise):
			curr = grouped[i]
			try:
				avg = np.median([x[1] for x in curr])
				avg2 = np.mean([x[1] for x in curr])
			except:
				avg = 0
			print("Average nFavs for level {} is {}".format(i, [avg, avg2]))

	def assignment_histogram(self):
		mylist = []
		for artist in self.artist_assignments:
			mylist.append(list(self.artist_assignments[artist].values()))

		mylist = [item for sublist in mylist for item in sublist]
		plt.hist(mylist)
		plt.show()

	def plot_user_expertise_progression(self):
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
			axes[i%10, int(i/10)].scatter(range(len(y)), ratings, c=vals)
			i += 1
			if i == 20: break
		plt.show()

if __name__ == '__main__':

	from data import Data 
	data = Data(False)
	fn = '../cache/VBPR_3_3_0.5_0.007_default_reg.pkl'
	avbpr = AVBPR(*data.get_max(), filename=fn, lr=0.0005, lr2=0.0005)
	valid_data = data.generate_evaluation_samples(True)
	assign_triples = data.generate_assignment_triples(1)
	avbpr.set_visual_data(data.get_visual_data())
	avbpr.set_dd_dict(data.get_dd_dict())
	avbpr.set_assignment_triples(assign_triples)
	avbpr.set_dicts(*data.get_artist_dicts())
	avbpr.initialize_assignments()
	for i in range(3):
		train_data = data.generate_train_samples(1000000)
		avbpr.train(train_data, valid_data, validation_freq=1000000)
		avbpr.assign_classes()
		avbpr.validation_aucs.append(avbpr.AUC(valid_data))

	import pdb; pdb.set_trace()
	avbpr.plot_validation_error()