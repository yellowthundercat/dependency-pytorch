import os

class Config:
	def __init__(self):
		# general
		self.model_name = 'test_model_biaffine'
		self.mode = 'train'  # option: 'train', 'evaluate', 'annotate'
		self.continue_train = False
		self.use_small_subset = True
		self.use_pos = True
		self.pos_type = 'vn'  # vn, uni, lab
		self.use_word_emb_scratch = False
		self.use_phobert = True
		self.use_charCNN = False
		self.cross_view = False

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
			train_data_folder = self.data_small_folder
			self.unlabel_folder = os.path.join(self.data_folder, 'unlabel_data_small')
		else:
			train_data_folder = os.path.join(self.data_folder, 'vndt')
			self.unlabel_folder = os.path.join(self.data_folder, 'unlabel_data')
		self.train_file = os.path.join(train_data_folder, 'train.txt')
		self.dev_file = os.path.join(train_data_folder, 'dev.txt')
		self.test_file = os.path.join(train_data_folder, 'test.txt')
		self.pos_train_file = os.path.join(train_data_folder, 'lab_train.txt')
		self.pos_dev_file = os.path.join(train_data_folder, 'lab_dev.txt')
		self.pos_test_file = os.path.join(train_data_folder, 'lab_test.txt')
		if self.pos_type == 'uni':
			self.pos_train_file = os.path.join(train_data_folder, 'uni_train.txt')
			self.pos_dev_file = os.path.join(train_data_folder, 'uni_dev.txt')
			self.pos_test_file = os.path.join(train_data_folder, 'uni_test.txt')

		# for annotation
		self.input_filename = 'input.txt'
		self.output_filename = 'output.txt'

		# word level
		self.concat_first_layer = False
		self.phobert_layer = -1  # range: [0, ..., 12]
		self.phobert_subword = 'first'  # sum or first
		self.fine_tune = True
		self.word_emb_dim = 100
		self.minimum_frequency = 2
		self.phobert_dim = 768
		self.pos_emb_dim = 100
		self.charCNN_dim = 0  # set later in code about 150

		# sentence level
		self.length_ordered = False
		self.teacher_dropout = 0.33
		self.student_dropout = 0.5
		self.arc_mlp_size = 500
		self.lab_mlp_size = 100

		# encoder
		self.encoder = 'biLSTM'  # biLSTM, transformer
		self.rnn_size = 400  # output encode = 4*rnn_size (2 biLSTM)
		self.rnn_1_depth = 3
		self.rnn_2_depth = 0
		self.transformer_1_depth = 2
		self.transformer_2_depth = 2
		self.transformer_dim = 128
		self.transformer_head = 4
		self.transformer_ff_dim = 256
		self.transformer_dropout = 0.2

		# train
		self.max_step = 20000
		self.max_waiting_step = 20000  # if not improve in this period -> stop
		self.teacher_only_step = 50
		self.batch_size = 32
		self.print_step = 50
		self.eval_dev_every = 500  # how often to evaluate on the dev set
		self.eval_test_every = 5000
		if self.use_small_subset:
			self.batch_size = 16
			self.print_step = 2
			self.eval_dev_every = 10
			self.eval_test_every = 30
			self.max_step = 100
			self.teacher_only_step = 0

		# cross-view
		self.gold_student_step = 10000000
		self.print_dev_student = True

		# optimizer
		# momentum for cross-view training
		self.use_momentum = False  # crossview should use
		if self.cross_view:
			self.use_momentum = True
		self.lr_momentum = 0.5  # base learning rate
		self.student_lr_momentum = 0.2
		self.pos_lambda = 1
		self.momentum = 0.9  # momentum
		self.grad_clip = 1.0  # maximum gradient norm during optimization
		self.warm_up_steps = 5000.0  # linearly ramp up the lr for this many steps
		self.lr_decay = 0.005  # factor for gradually decaying the lr

		# other
		self.seed = 1234
		self.error_order = False
