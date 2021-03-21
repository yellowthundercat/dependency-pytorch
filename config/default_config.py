import os

class Config:
	def __init__(self):
		# general
		self.model_name = 'test_model_cross_view'
		self.mode = 'train'  # option: 'train', 'evaluate', 'annotate'
		self.use_small_subset = True
		self.use_pos = True
		self.cross_view = True
		self.use_proccessed_embedding = True

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
		self.max_step = 10000
		self.batch_size = 32
		self.phobert_batch_size = 4
		self.print_step = 50
		self.eval_dev_every = 500  # how often to evaluate on the dev set
		if self.use_small_subset:
			self.print_step = 2
			self.eval_dev_every = 10
			self.max_step = 100

		# optimizer
		# momentum for cross-view training
		self.use_momentum = True
		self.lr_momentum = 0.5  # base learning rate
		self.momentum = 0.9  # momentum
		self.grad_clip = 1.0  # maximum gradient norm during optimization
		self.warm_up_steps = 5000.0  # linearly ramp up the lr for this many steps
		self.lr_decay = 0.005  # factor for gradually decaying the lr

		# adam for normal

		# other
		self.seed = 2712021
