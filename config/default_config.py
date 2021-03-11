import os

class Config:
	def __init__(self):
		# general
		self.model_name = 'test_model'
		self.mode = 'train'  # option: 'train', 'evaluate', 'annotate'
		self.use_small_subset = True

		# file location
		self.data_folder = 'data'
		self.corpus_file = os.path.join(self.data_folder, 'corpus.pickle')
		self.parsing_file = os.path.join(self.data_folder, 'parsing.txt')
		self.save_folder = os.path.join(self.data_folder, self.model_name)
		self.model_file = os.path.join(self.save_folder, 'model.pt')
		self.config_file = os.path.join(self.save_folder, 'config.pickle')
		self.vocab_file = os.path.join(self.save_folder, 'vocab.pickle')
		if self.use_small_subset:
			self.train_file = os.path.join(self.data_folder, 'small_train.txt')
			self.dev_file = os.path.join(self.data_folder, 'small_dev.txt')
			self.test_file = os.path.join(self.data_folder, 'small_test.txt')
		else:
			self.train_file = os.path.join(self.data_folder, 'train.txt')
			self.dev_file = os.path.join(self.data_folder, 'dev.txt')
			self.test_file = os.path.join(self.data_folder, 'test.txt')
		# for annotation
		self.input_filename = 'input.txt'
		self.output_filename = 'output.txt'

		# word level
		self.use_vn_pos = True
		self.word_emb_dim = 30
		self.phobert_dim = 768
		self.pos_emb_dim = 30

		#sentence level
		self.length_ordered = False
		self.drop_out_rate = 0.25
		self.rnn_size = 256
		self.rnn_depth = 3
		self.mlp_size = 256

		# train
		self.epoch = 30
		self.batch_size = 128
		self.phobert_batch_size = 4

		# other
		self.seed = 2712021
