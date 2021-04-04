import torch
from models.encoder import RNNEncoder

from models.parser import Parser
from models.deep_biaffine import DeepBiaffine
from utils import optimizer

def init_model_student(main_self, config):
	main_self.model_students = [
		Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_bi', 'uni_bi', config.student_dropout),
		Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_fw', 'uni_fw', config.student_dropout),
		Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_fw', 'uni_bw', config.student_dropout),
		Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_bw', 'uni_fw', config.student_dropout),
		Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_bw', 'uni_bw', config.student_dropout)
	]
	main_self.optimizer = optimizer.momentum(main_self.model.parameters(), config)
	main_self.optimizer_students = [optimizer.momentum(model.parameters(), config) for model in main_self.model_students]
	main_self.scheduler = optimizer.momentum_scheduler(main_self.optimizer, config)
	main_self.scheduler_students = [optimizer.momentum_scheduler(opt_student, config) for opt_student in
																	main_self.optimizer_students]

def init_model(main_self, config):
	main_self.encoder = RNNEncoder(config, len(main_self.corpus.vocab.t2i))
	if config.use_first_layer:
		main_self.model = Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_bi', 'uni_bi', config.teacher_dropout)
	else:
		main_self.model = Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'bi', 'bi', config.teacher_dropout)
	main_self.saving_step = 0
	main_self.best_las = main_self.best_uas = 0
	if config.cross_view:
		init_model_student(main_self, config)
	else:
		# main_self.optimizer = optimizer.adam(main_self.model.parameters(), config)
		if config.use_momentum:
			main_self.optimizer = optimizer.momentum(main_self.model.parameters(), config)
			main_self.scheduler = optimizer.momentum_scheduler(main_self.optimizer, config)
		else:
			main_self.optimizer = optimizer.adam(main_self.model.parameters(), config)

def load_model(main_self, all_model, config):
	main_self.encoder = all_model['encoder']
	main_self.model = all_model['model']
	main_self.model.encoder = main_self.encoder
	main_self.optimizer = all_model['optimizer']
	if config.use_momentum:
		main_self.scheduler = all_model['scheduler']
	main_self.saving_step = all_model['step']
	main_self.best_uas = all_model['uas']
	main_self.best_las = all_model['las']
	if config.cross_view:
		main_self.model_students = all_model['model_students']
		for student_model in main_self.model_students:
			student_model.encoder = main_self.encoder
		main_self.optimizer_students = all_model['optimizer_students']
		main_self.scheduler_students = all_model['scheduler_students']
		# main_self.scheduler = all_model['scheduler']

def save_model(main_self, config):
	all_model = {
		'encoder': main_self.encoder,
		'model': main_self.model,
		'optimizer': main_self.optimizer,
		'step': main_self.saving_step,
		'uas': main_self.best_uas,
		'las': main_self.best_las
	}
	if config.use_momentum:
		all_model['scheduler'] = main_self.scheduler
	if config.cross_view:
		all_model['model_students'] = main_self.model_students
		all_model['optimizer_students'] = main_self.optimizer_students
		all_model['scheduler_students'] = main_self.scheduler_students
	torch.save(all_model, main_self.config.model_file)
