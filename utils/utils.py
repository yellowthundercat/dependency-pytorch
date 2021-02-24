import os
import sys

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



