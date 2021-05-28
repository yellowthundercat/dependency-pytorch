import torch
from models.encoder import Encoder

from models.parser import Parser
from utils import optimizer

def init_model_student(main_self, config, get_optimizer):
	teacher_encoder = 'bi'
	if config.concat_first_layer:
		teacher_encoder = 'uni_bi'
	if config.encoder == 'biLSTM':
		main_self.model_students = [
			Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, teacher_encoder, teacher_encoder, config.student_dropout),
			Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_fw', 'uni_fw', config.student_dropout),
			Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_fw', 'uni_bw', config.student_dropout),
			Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_bw', 'uni_fw', config.student_dropout),
			Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_bw', 'uni_bw', config.student_dropout)
		]
	else:
		main_self.model_students = [
			Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'trans1', 'trans1', config.student_dropout),
		]
	main_self.optimizer, main_self.scheduler = get_optimizer(main_self.model, config, 'teacher')
	main_self.optimizer_students = []
	main_self.scheduler_students = []
	for model in main_self.model_students:
		opt, sche = get_optimizer(model, config, 'student')
		main_self.optimizer_students.append(opt)
		main_self.scheduler_students.append(sche)

def init_model(main_self, config):
	get_optimizer = optimizer.adamW
	if config.use_momentum:
		get_optimizer = optimizer.momentum
	main_self.encoder = Encoder(config, len(main_self.corpus.vocab.t2i), len(main_self.corpus.vocab.w2i), len(main_self.corpus.vocab.c2i))
	if config.concat_first_layer:
		main_self.model = Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'uni_bi', 'uni_bi', config.teacher_dropout)
	else:
		main_self.model = Parser(main_self.encoder, len(main_self.corpus.vocab.l2i), config, 'bi', 'bi', config.teacher_dropout)
	main_self.saving_step = 0
	main_self.best_las = main_self.best_uas = 0
	main_self.using_amsgrad = False
	main_self.best_loss = 100
	if config.cross_view:
		init_model_student(main_self, config, get_optimizer)
	else:
		main_self.optimizer, main_self.scheduler = get_optimizer(main_self.model, config, 'teacher')

def load_model(main_self, all_model, config):
	main_self.encoder = all_model['encoder']
	main_self.model = all_model['model']
	main_self.model.encoder = main_self.encoder
	main_self.optimizer = all_model['optimizer']
	main_self.scheduler = all_model['scheduler']
	main_self.saving_step = all_model['step']
	main_self.best_uas = all_model['uas']
	main_self.best_las = all_model['las']
	main_self.best_loss = all_model['loss']
	main_self.using_amsgrad = all_model['using_amsgrad']
	if config.cross_view:
		main_self.model_students = all_model['model_students']
		for student_model in main_self.model_students:
			student_model.encoder = main_self.encoder
		main_self.optimizer_students = all_model['optimizer_students']
		main_self.scheduler_students = all_model['scheduler_students']

def save_model(main_self, config):
	all_model = {
		'encoder': main_self.encoder,
		'model': main_self.model,
		'optimizer': main_self.optimizer,
		'step': main_self.saving_step,
		'uas': main_self.best_uas,
		'las': main_self.best_las,
		'loss': main_self.best_loss,
		'scheduler': main_self.scheduler,
		'using_amsgrad': main_self.using_amsgrad
	}
	if config.cross_view:
		all_model['model_students'] = main_self.model_students
		all_model['optimizer_students'] = main_self.optimizer_students
		all_model['scheduler_students'] = main_self.scheduler_students
	torch.save(all_model, main_self.config.model_file)
