import os
import sys
import matplotlib.pyplot as plt

def ensure_dir(d, alert=True):
	if not os.path.exists(d):
		if alert:
			print("Directory {} do not exist; creating".format(d))
		os.makedirs(d)

# print log
def log(*args):
	msg = ' '.join(map(str, args))
	sys.stdout.write(msg + '\n')
	sys.stdout.flush()

def heading(*args):
	log()
	log(80 * '=')
	log(*args)
	log(80 * '=')

def show_history_graph(history):
	plt.plot(history['train_loss'])
	plt.plot(history['val_loss'])
	plt.plot(history['uas'])
	plt.legend(['training loss', 'validation loss', 'UAS'])
	plt.show()

