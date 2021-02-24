import os

class Config:
	def __init__(self):
		# general
		self.model_name = 'test_model'
		self.use_cuda = False
		self.mode = 'train'  # option: 'train', 'evaluate', 'annotate'

		# file location
		self.data_folder = 'data'
		self.save_folder = os.path.join(self.data_folder, self.model_name)
		self.model_file = os.path.join(self.save_folder, 'model.pt')
		self.train_file = os.path.join(self.data_folder, 'train.txt')
		self.dev_file = os.path.join(self.data_folder, 'dev.txt')
		self.test_file = os.path.join(self.data_folder, 'test.txt')
		# for annotation
		self.input_filename = 'input.txt'
		self.output_filename = 'output.txt'

		# train
		self.epoch = 1
		self.batch_size = 32

		# other
		self.seed = 2712021
