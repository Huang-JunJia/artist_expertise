from collections import OrderedDict

import pickle as pkl
import numpy as np
import sys


class Data:

	def __init__(self, create_test=True):
		self.data = pkl.load(open('../data/favorited_dict.p', 'rb'))
		self.max_item, self.max_user = pkl.load(
			open('../data/useful_stats.p', 'rb'))
		self.visual_data = pkl.load(
			open('../data/id_feature_dict_with_artist.p', 'rb'))
		self.artist_dict = pkl.load(open('../data/artist_dict.p', 'rb'))
		self.item_to_artist = pkl.load(open('../data/item_to_artist.p', 'rb'))
		self.img_nfavs_dict = pkl.load(open('../data/img_nfavs.p', 'rb'))
		self.dd_dict = pkl.load(open('../data/dd_dict.p', 'rb'))
		self.time_dict = pkl.load(open('../data/time_dict.p', 'rb'))
		self.__clean_data()
		self.__create_dict(create_test)

	def __clean_data(self):
		#-------- Get a list of items that were removed due to no image -----#
		removed_items = set()
		for key in range(self.max_item):
			if key not in self.visual_data:
				removed_items.add(key)
			else:
				if not isinstance(self.visual_data[key][0], np.ndarray) \
				or not self.time_dict[key]: 
					removed_items.add(key) 
		self.removed_items = removed_items

		#--------- Clean up artist dict -------------------------#
		self.__clean_artist_dict()

		#--------- Clean up dataset after removing items --------#
		data = self.data

		for user in data:
			rated_items = data[user]
			removed = []
			for item in rated_items:
				if item in removed_items:
					removed.append(item)
			for item in removed:
				rated_items.remove(item)
			data[user] = rated_items


		users_to_remove = []
		for user in data:
			images = np.unique(data[user]).tolist()
			images = [image for image in images if image in self.visual_data]
			if not images: users_to_remove.append(user)
			else: data[user] = images

		for user in users_to_remove:
			data.pop(user)

		self.data = data

		#--------- Correct the formatting ---------#
		for key in self.visual_data:
			self.visual_data[key] = self.visual_data[key][0]



	def __clean_artist_dict(self):

		#--------- Remove duplicates, but maintain order --------#
		for key, value in list(self.artist_dict.items()):
			self.artist_dict[key] = list(OrderedDict.fromkeys(value))
		
		#--------- Remove deleted items from artist's list --------#
		for item in self.removed_items:
			artist = self.item_to_artist[item]
			self.artist_dict[artist].remove(item)

		#--------- Remove empty artists from dict --------#
		for key, value in list(self.artist_dict.items()):
			if not value:
				self.artist_dict.pop(key)


	def __create_dict(self, create_test):
		data = self.data 

		valid = {}; test = {}; train = {}

		if create_test: 
			for key in data:
				rated_items = data[key]
				if len(rated_items) > 2:
					items =  np.random.choice(rated_items, 2, replace=False)
					valid[key] = items[0]
					test[key] = items[1]
					train[key] = [item for item in rated_items if item not in items]

		else:
			for key in data:
				rated_items = data[key]
				if len(rated_items) > 1:
					item = np.random.choice(rated_items)
					rated_items.remove(item)
					valid[key] = item
					train[key] = rated_items

		self.train_dict = train
		self.valid_dict = valid
		self.test_dict = test 



	def generate_train_samples(self, nSamples=1):
		
		#-------- Create local variables for convenience and brevity ------#
		train = self.train_dict
		valid = self.valid_dict
		test = self.test_dict

		removed_items = self.removed_items
		max_item = self.max_item
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

			unrated_item = np.random.choice(max_item)
			while unrated_item in rated_items or unrated_item in removed_items \
				or unrated_item == valid_item or unrated_item == test_item:
				unrated_item = np.random.choice(max_item)

			rated_item = np.random.choice(rated_items)

			samples[i, 1] = rated_item
			samples[i, 2] = unrated_item

		return samples


	def generate_evaluation_samples(self, isValid=True, nItems = 5):
		
		#-------- Create local variables for convenience and brevity ------#
		train = self.train_dict
		valid = self.valid_dict
		test = self.test_dict

		removed_items = self.removed_items
		max_item = self.max_item
		#------------------------------------------------------------------#

		if isValid: item_list = list(valid.items())
		else: item_list = list(test.items())

		users = [x[0] for x in item_list if x[1] is not None]
		users = np.repeat(users, nItems)
		
		items = [x[1] for x in item_list if x[1] is not None]
		items = np.repeat(items, nItems)

		samples = np.zeros((len(users), 3), dtype=int)
		samples[:, 0] = users
		samples[:, 1] = items

		for i, user in enumerate(users):
			rated_items = train[user]
			valid_item = valid[user]
			if test: test_item = test[user]
			else: test_item = None

			unrated_item = np.random.choice(max_item)
			while unrated_item in rated_items or unrated_item in removed_items or unrated_item == valid_item:
				unrated_item = np.random.choice(max_item)

			samples[i, 2] = unrated_item

		return samples

	def get_max(self):
		return self.max_item, self.max_user

	def get_visual_data(self):
		return self.visual_data

	def get_dd_dict(self):
		return self.dd_dict

	def get_artist_dicts(self):
		return self.artist_dict, self.item_to_artist, self.img_nfavs_dict



	'''
	Creates a list of triples which will be used to get the objective for 
	each artist and therefore find the set of best assignments.
	'''
	def generate_assignment_triples(self, batch_size=5):

		#-------- Create local variables for convenience and brevity ------#
		train = self.train_dict
		valid = self.valid_dict
		test = self.test_dict

		removed_items = self.removed_items
		max_item = self.max_item
		#------------------------------------------------------------------#

		keys = list(train.keys())
		vals = [train[key] for key in keys]
		all_items = [item for sublist in vals for item in sublist]
		users_list = [[key]*len(train[key]) for key in keys]
		all_users = [item for sublist in users_list for item in sublist]
		samples = np.zeros((len(all_items)*batch_size, 3), dtype=int)

		samples[:,0] =  np.repeat(all_users, batch_size)
		samples[:,1] =  np.repeat(all_items, batch_size)

		for i, user in enumerate(all_users):
			rated_items = train[user]
			valid_item = valid[user]

			if test: test_item = test[user]
			else: test_item = None

			unrated_item = np.random.choice(max_item)
			while unrated_item in rated_items or unrated_item in removed_items \
				or unrated_item == valid_item or unrated_item == test_item:
				unrated_item = np.random.choice(max_item)
				samples[i,2] = unrated_item

		return samples


