import os

class Config:
	def __init__(self):
		# general
		self.model_name = 'test_model'
		self.mode = 'train'  # option: 'train', 'evaluate', 'annotate'
		self.continue_train = False
		self.use_small_subset = True
		self.use_pos = True
		self.use_phobert = True
		self.use_charCNN = True
		self.cross_view = True

		# file location
		self.data_folder = 'data'
		self.data_small_folder = os.path.join(self.data_folder, 'data_small')
		self.parsing_file = os.path.join(self.data_folder, 'parsing.txt')
		self.annotate_file = os.path.join(self.data_folder, 'annotate.txt')
		self.error_sample_file = os.path.join(self.data_folder, 'error_sample.txt')
		self.save_folder = os.path.join(self.data_folder, self.model_name)
		self.model_file = os.path.join(self.save_folder, 'all_model.pt')
		self.config_file = os.path.join(self.save_folder, 'config.pickle')
		self.vocab_file = os.path.join(self.save_folder, 'vocab.pickle')
		if self.use_small_subset:
			data_folder = self.data_small_folder
			self.unlabel_folder = os.path.join(self.data_folder, 'unlabel_data_small')
		else:
			data_folder = self.data_folder
			self.unlabel_folder = os.path.join(self.data_folder, 'unlabel_data')
		self.train_file = os.path.join(data_folder, 'train.txt')
		self.dev_file = os.path.join(data_folder, 'dev.txt')
		self.test_file = os.path.join(data_folder, 'test.txt')
		self.new_train_file = os.path.join(data_folder, 'new_train.txt')
		self.new_dev_file = os.path.join(data_folder, 'new_dev.txt')
		self.new_test_file = os.path.join(data_folder, 'new_test.txt')
		# for annotation
		self.input_filename = 'input.txt'
		self.output_filename = 'output.txt'

		# word level
		self.use_first_layer = False
		self.phobert_layer = 8  # range: [0, ..., 12]
		# attention requires format: [(a,b), (a,b)] with a is hidden layer, b is head, if b is '*' = get all
		# range: [(0..11, 0..11 or *)]
		# self.attention_requires = [(7, '*'), (8, '*')]
		# self.attention_head_tops = 2
		self.pos_type = 'lab'  # vn, ud, lab
		self.word_emb_dim = 100
		self.phobert_dim = 768
		self.pos_emb_dim = 50
		self.charCNN_dim = 0  # set later in code

		#sentence level
		self.length_ordered = False
		self.teacher_dropout = 0.33
		self.student_dropout = 0.5
		self.rnn_size = 200  # output encode = 4*rnn_size (2 biLSTM)
		self.rnn_depth = 3
		self.arc_mlp_size = 200
		self.lab_mlp_size = 100

		# train
		self.max_step = 20000
		self.batch_size = 64
		self.print_step = 50
		self.eval_dev_every = 500  # how often to evaluate on the dev set
		if self.use_small_subset:
			self.batch_size = 16
			self.print_step = 2
			self.eval_dev_every = 10
			self.max_step = 100

		# optimizer
		# momentum for cross-view training
		self.use_momentum = False  # crossview must use
		self.lr_momentum = 0.5  # base learning rate
		self.student_lr_momentum = 0.2
		self.momentum = 0.9  # momentum
		self.grad_clip = 1.0  # maximum gradient norm during optimization
		self.warm_up_steps = 5000.0  # linearly ramp up the lr for this many steps
		self.lr_decay = 0.005  # factor for gradually decaying the lr

		# adam for normal

		# other
		self.seed = 2712021
