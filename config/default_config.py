import os

class Config:
	def __init__(self):
		# general
		self.model_name = 'test_model_cross_view'
		self.mode = 'train'  # option: 'train', 'evaluate', 'annotate'
		self.use_small_subset = False
		self.use_pos = False

		# file location
		self.data_folder = 'data'
		self.corpus_file = os.path.join(self.data_folder, 'corpus.pickle')
		self.parsing_file = os.path.join(self.data_folder, 'parsing.txt')
		self.annotate_file = os.path.join(self.data_folder, 'annotate.txt')
		self.error_sample_file = os.path.join(self.data_folder, 'error_sample.txt')
		self.save_folder = os.path.join(self.data_folder, self.model_name)
		self.model_file = os.path.join(self.save_folder, 'all_model.pt')
		self.config_file = os.path.join(self.save_folder, 'config.pickle')
		self.vocab_file = os.path.join(self.save_folder, 'vocab.pickle')
		self.unlabel_embedding_folder = os.path.join(self.data_folder, 'unlabel_embedding')
		if self.use_small_subset:
			self.train_file = os.path.join(self.data_folder, 'small_train.txt')
			self.dev_file = os.path.join(self.data_folder, 'small_dev.txt')
			self.test_file = os.path.join(self.data_folder, 'small_test.txt')
			self.unlabel_folder = os.path.join(self.data_folder, 'unlabel_data_small')
		else:
			self.train_file = os.path.join(self.data_folder, 'train.txt')
			self.dev_file = os.path.join(self.data_folder, 'dev.txt')
			self.test_file = os.path.join(self.data_folder, 'test.txt')
			self.unlabel_folder = os.path.join(self.data_folder, 'unlabel_data')
		# for annotation
		self.input_filename = 'input.txt'
		self.output_filename = 'output.txt'

		# word level
		self.use_vn_pos = True
		self.word_emb_dim = 100
		self.phobert_dim = 768
		self.pos_emb_dim = 50

		#sentence level
		self.length_ordered = False
		self.input_dropout = 0.33
		self.word_dropout = 0.33
		self.pos_dropout = 0.33
		self.rnn_dropout = 0.33
		self.arc_mlp_dropout = 0.33
		self.lab_mlp_dropout = 0.33
		self.rnn_size = 200  # output encode = 4*rnn_size (2 biLSTM)
		self.rnn_depth = 3
		self.arc_mlp_size = 200
		self.lab_mlp_size = 100

		# train
		self.epoch = 25
		self.batch_size = 32
		self.phobert_batch_size = 4

		# other
		self.seed = 2712021
