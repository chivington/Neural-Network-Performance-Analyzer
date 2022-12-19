import numpy as np
import requests as rq
import matplotlib.pyplot as plt
import time, sys, os, json


# #################### DON'T CHANGE ANYTHING ABOVE THIS LINE ####################
import cupy as cu	# <-- COMMENT THIS LINE!  ...if you don't have CuPy installed

# ----- SELECT MATH ENVIRONMENT
MATH_ENV = 'numpy'							# To switch between NumPy & CuPy, change this to 'numpy' or 'cupy'
blas = np if MATH_ENV == 'numpy' else cu	# DON'T CHANGE THIS LINE!!!

# ----- TURN ON/OFF ALL PRINTING
# Choose between: 'ALL', 'MODEL_ONLY', 'NO_MODEL', or 'NONE'
# - ALL: prints info related to model training/testing performance, and data loading/processing, etc.
# - MODEL_ONLY: prints info related to model training/testing performance only
# - NO_MODEL: prints info related to data loading/processing, etc. only
# - NONE: only prints when writing to a file
PRINTING = 'ALL'
# #################### DON'T CHANGE ANYTHING BELOW THIS LINE ####################


MODEL_PRINTING = True if PRINTING == 'ALL' or PRINTING == 'MODEL_ONLY' else False
PROGRAM_PRINTING = True if PRINTING == 'ALL' or PRINTING == 'NO_MODEL' else False

# ----- SEED RNG
blas.random.seed(4)


# ----- FORMATTING UTILS
def clear_screen():
	if os.name == 'nt': os.system('cls')
	else: os.system('clear')

def progress(i, total):
	str = f'({round(i/total * 100, 2)}%)   '
	n = len(str)
	for j in range(n): sys.stdout.write(f' ')
	for k in range(n): sys.stdout.write(f'\b')
	sys.stdout.write(f'{str}')
	for k in range(n): sys.stdout.write(f'\b')
	sys.stdout.flush()

def greet(clear=False):
	if clear: clear_screen()
	if PROGRAM_PRINTING: print((
		"\n CuPy GPU-Powered MNIST Neural Network \n"
		" ------------------------------------- \n"
		"  This neural network uses your GPU to train on the MNIST \n"
		" dataset and learn to recognize images of hand-written digits. \n\n"
		"  If you don't have a GPU or don't have CuPy installed on \n"
		" your system, simply change the MATH_ENV variable to use NumPy \n"
		" instead. The training will run much slower, but it will \n"
		" achieve the same results. \n\n"
		" Email john@discefasciendo.com with questions.\n\n Enjoy! \n\n\n"
	))
	if not PROGRAM_PRINTING: print(f' Program printing turned off.')
	if not MODEL_PRINTING: print(f' Model printing turned off.')


# ----- DATA UTILS
def one_hot(Y, classes):
	encoded = blas.zeros((Y.shape[0], classes))
	for i in range(Y.shape[0]):
		encoded[i][int(Y[i][0])] = 1
	return encoded

def normalize(x):
	x = x / x.max()
	return x

def shuffle(X, Y):
	idxs = blas.array([i for i in range(X.shape[0])])
	blas.random.shuffle(idxs)
	return X[idxs], Y[idxs]

def load_data():
	if PROGRAM_PRINTING: print(f' PRE-PROCESSING DATA...\n Loading training & testing datasets...')
	files = ['mnist_train', 'mnist_test']
	out = []
	for file in files:
		if PROGRAM_PRINTING: sys.stdout.write(f'  - {file}')
		load_start = time.time()
		data = np.loadtxt(f'datasets/{file}.csv', delimiter = ',')
		x = normalize(data[:,1:])
		y = one_hot(data[:,:1], 10)
		if MATH_ENV == 'cupy':
			x, y = cu.array(x), cu.array(y)
		load_end = time.time()
		out.append((x, y))
		if PROGRAM_PRINTING: print(f' ({round(load_end - load_start, 2)}s)')
	if PROGRAM_PRINTING: print('')
	return [out[0][0], out[0][1], out[1][0], out[1][1]]

def batch_data(X, Y, batch_size, cycles):
	if PROGRAM_PRINTING: sys.stdout.write(f' Batching training dataset ({batch_size})... ')
	batching_start = time.time()
	train_batches = []
	for e in range(cycles):
		if PROGRAM_PRINTING: progress(e, cycles)
		shuffled_X, shuffled_Y = shuffle(X, Y)

		m = X.shape[0]
		num_batches = m // batch_size

		batches = []
		for batch in range(num_batches - 1):
			start = batch * batch_size
			end = (batch + 1) * batch_size
			x = X[start:end]
			y = Y[start:end]
			batches.append((x, y))

		last_start = num_batches * batch_size
		batches.append((X[last_start:], Y[last_start:]))

		train_batches.append(batches)
	batching_end = time.time()
	if PROGRAM_PRINTING: print(f'({blas.around(batching_end - batching_start, 2)}s)    ')
	return train_batches

def download_data():
	if PROGRAM_PRINTING: print(' Downloading MNIST data...')
	train_url = 'https://pjreddie.com/media/files/mnist_train.csv'
	test_url = 'https://pjreddie.com/media/files/mnist_test.csv'
	for url in [train_url, test_url]:
		f = url.split('/')[-1]
		if PROGRAM_PRINTING: sys.stdout.write(f'  - {f}')
		start = time.time()
		req = rq.get(url)
		res = req.text
		fp = open(f'data/{f}', 'w')
		fp.write(res)
		fp.close
		end = time.time()
		if PROGRAM_PRINTING: sys.stdout.write(f' ({blas.around(end-start, 2)}s)\n')

# ----- METRICS UTILS
def plot_lines(models, model_idx, model_acc):
	title = f'MNIST Network Performance Metrics'
	fig, plots = plt.subplots(3, figsize=(9,6))
	plt.suptitle(f'{title}\n Stats vs. Cycles\n (Best Model: #{model_idx} - Acc. {model_acc}%)', fontsize=16, fontweight='bold')
	fig.subplots_adjust(top=0.85)

	for i, p in enumerate(plots):
		for m, model in enumerate(models):
			test_acc, data = model[0], model[1]
			plot_data = data[i]["data"]
			plot_title = data[i]["title"]

			if MATH_ENV == 'cupy':
				plot_data = plot_data.get()

			lbls = [
				f'{blas.around(plot_data[0] - plot_data[-1], 4)} decrease',
				f'{test_acc}% acc.',
				f'{blas.around(blas.sum(plot_data), 2)}s'
			]
			p.plot(range(1, len(plot_data) + 1), plot_data, label=f'{m+1}: {lbls[i]}')
			p.legend(loc='lower right' if plot_title == 'Accuracy' else 'upper right')
			p.set(ylabel = plot_title)
	plt.show()

def show_predictions(test_imgs, predictions, model_idx, model_acc):
	img_count = 9
	idxs = blas.random.randint(0, test_imgs.shape[0], size=img_count)
	imgs = test_imgs[idxs]
	pixels = int(blas.sqrt(imgs.shape[1]))
	if MATH_ENV == 'cupy': imgs = imgs.get()
	imgs = imgs.reshape([img_count, pixels, pixels])
	preds = blas.argmax(predictions[idxs], axis=1)
	pred_count = len(preds)
	rows = int(blas.sqrt(pred_count) // 1)
	cols = rows if rows**2 == pred_count else (rows + 1)
	fig, axs = plt.subplots(rows, cols)
	plt.suptitle(f' Best MNIST Model Predictions\n (Model #{model_idx} - Acc. {model_acc}%)', fontsize=16, fontweight='bold')
	for row in range(rows):
		for col in range(cols):
			i = row * rows + col
			axs[row, col].set_title(f'Prediction: {preds[i]}', color='black')
			axs[row, col].imshow(imgs[i], interpolation='nearest')
	plt.tight_layout()
	plt.show()
	print('\n')

def record_performances(metrics, fn='performance'):
	print(f' Writing results to file: metrics/{fn}.txt')
	if not os.path.exists(f'metrics/{fn}.txt'):
		fp = open(f'metrics/{fn}.txt', 'w')
		fp.write(f'env, dims, cycles, learning_rate, batch_size, total_train_time, avg_cycle_time, training_accuracy, test_accuracy\n')
		fp.close()
	fp = open(f'metrics/{fn}.txt', 'a')
	for line in metrics:
		fp.write(f'{MATH_ENV}, {line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}, {line[5]}, {line[6]}, {line[7]}\n')
	fp.close()

def save_model_weights(model, metrics, fn='model-weights'):
	print(f' Saving best model weights to file: weights/{fn}.txt')
	fp = open(f'weights/{fn}.txt', 'w')
	fp.write(f'env, dims, cycles, learning_rate, batch_size, total_train_time, avg_cycle_time, training_accuracy, test_accuracy\n')
	fp.write(f'{MATH_ENV}, {metrics[0]}, {metrics[1]}, {metrics[2]}, {metrics[3]}, {metrics[4]}, {metrics[5]}, {metrics[6]}, {metrics[7]}\n')
	for l,layer in enumerate(model.layers):
		if layer.type == 'Dense':
			lidx = 1 if l == 0 else int(l // 2) + 1
			fp.write(f'\nLAYER {lidx} WEIGHTS {layer.weights.shape}\n')
			for row in layer.weights:
				for c,col in enumerate(row):
					fp.write(f'{row[c]}{"," if c<layer.weights.shape[1]-1 else ""}')
				fp.write('\n\n')
	fp.close()

def load_model_weights(filename):
	if PROGRAM_PRINTING: print(f' Loading previous model weights for: {filename}')
	parse_dims = lambda str: np.array(str[:-2].split('(')[1].split(', '), dtype=int)
	parse_weights = lambda str: np.array(str[:-1].split(','), dtype=np.float64)

	fp = open(f'weights/{filename}.txt', 'r')
	lines = fp.readlines()[3:]
	current_line = 0
	dims = []
	weights = []
	while current_line < len(lines):
		dims = parse_dims(lines[current_line])
		weights.append(np.zeros(dims))

		for row in range(dims[0]):
			current_line = current_line + 1
			parsed_line = parse_weights(lines[current_line])
			for col in range(dims[1]):
				weights[-1][row,col] = parsed_line[col]
			current_line = current_line + 1

		current_line = current_line + 2

	return weights if MATH_ENV == 'numpy' else [cu.array(w) for w in weights]


# ----- TESTING UTILS
def train_models(datasets, models):
	train_x, train_y, test_x, test_y = datasets
	metrics, performance_data = [], []
	best_idx, best_acc, best_preds, best_model = 0, 0.0, None, None

	for i,model in enumerate(models):
		if PROGRAM_PRINTING: print(f'\n\n TESTING MODEL {i+1}...')

		# get model hyperparameters
		dims = model['dims']
		cycles = model['cycles']
		bs = model['bs']
		lr = model['lr']
		pf = model['pf']

		# Batch training data preemptively to speed up th etraining process.
		train_batches = batch_data(train_x, train_y, bs, cycles)

		# train & test model
		m = Net(train_x, train_y, dims, cycles, lr, pf)
		costs, accs, times, train_time, avg_time = m.train(train_batches)
		test_acc, predictions = m.test(test_x, test_y)

		# save model parameters & performance
		if not isinstance(dims[0], int): dims = [d.shape[0] for d in dims]
		metrics.append([dims, cycles, lr, bs, train_time, avg_time, accs[-1], test_acc])
		performance_data.append((test_acc, [
			{'title': 'Cost', 'data': costs},
			{'title': 'Accuracy', 'data': accs},
			{'title': 'Time', 'data': times}
		]))
		if test_acc > best_acc or i == 0:
			best_idx = i
			best_model = m
			best_acc = test_acc
			best_preds = predictions
		if PROGRAM_PRINTING: print('')

	return best_idx+1, best_model, best_preds, best_acc, metrics, performance_data

def evaluate_models(datasets, models, record_performance, save_weights, logfile):
	# test models & get metrics/performance data
	best_idx, best_model, best_preds, best_acc, metrics, performance_data = train_models(datasets, models)

	# show predictions of best model
	show_predictions(datasets[2], best_preds, best_idx, best_acc)

	# plot model performance & write to file
	plot_lines(performance_data, best_idx, best_acc)

	# optionally, save performance metrics of all models tested
	if record_performance: record_performances(metrics, logfile)

	# optionally, save weights of the best performing model
	if save_weights: save_model_weights(best_model, metrics[best_idx-1], logfile)


# ----- NEURAL NETWORK CLASSES
class Net:
	def __init__(self, X, Y, layers=[256,128], cycles=3, lr=0.001, print_freq=10):
		if MODEL_PRINTING: print(f'\n Initializing network... (cycles={cycles}, learning rate={lr})')
		self.input = X
		self.labels = Y
		self.layers = self.init_layers(layers)
		self.cycles = cycles
		self.lr = lr
		self.print_freq = print_freq

	def init_layers(self, layers):
		init = []
		num_layers = len(layers)

		if not isinstance(layers[0], int):
			for l,layer in enumerate(layers):
				init.append(Dense(layer))
				init.append(Softplus() if l < (num_layers-1) else Softmax())
				if MODEL_PRINTING:
					print(f'  Layer {l+1} Dimensions: ({layer.shape[0]} x {layer.shape[1]})')
		else:
			n = self.input.shape[1]
			for l in range(num_layers + 1):
				if l < num_layers:
					layer = layers[l]
					input_size = n if l == 0 else layers[l-1]
					output_size = layer
					init.append(Dense([input_size, output_size]))
					init.append(Softplus())
				else:
					init.append(Dense([layers[-1], 10]))
					init.append(Softmax())
				if MODEL_PRINTING:
					sizes = init[-2].weights.shape
					print(f'  Layer {l+1} Dimensions: ({sizes[0]} x {sizes[1]})')

		return init

	def forward(self, batch):
		output = batch
		for layer in self.layers:
			output = layer.forward(output)
		return output

	def backward(self, batch, error):
		grad = (1/batch.shape[0]) * error # init to cost_grad
		for layer in list(reversed(self.layers)):
			grad = layer.backward(grad, self.lr)

	def train(self, batches):
		if MODEL_PRINTING: print(f'\n TRAINING...')
		m, n = self.input.shape
		costs, accs, times = blas.array([]), blas.array([]), blas.array([])
		batch_count = len(batches[0])
		batch_size = batches[0][0][0].shape[0]

		train_start = time.time()
		for cycle in range(self.cycles):
			current_batches = batches[cycle]
			cycle_start = time.time()
			cost, acc = 0, 0
			print_cycle = True if MODEL_PRINTING and (cycle % self.print_freq == 0) else False

			if print_cycle:
				sys.stdout.write(f'   CYCLE {f"{cycle + 1}/{self.cycles}":<5} >> ')

			for b,batch in enumerate(current_batches):
				if print_cycle: progress(b, batch_count)
				if cycle==0 and b==13:
					cycle_start = time.time()
					train_start = time.time()
				output = self.forward(batch[0])
				error = output - batch[1]
				self.backward(batch[0], error)
				cost += blas.mean(error**2)
				acc += blas.count_nonzero(blas.argmax(output, axis=1) == blas.argmax(batch[1], axis=1)) / batch[0].shape[0]

			cycle_end = time.time()
			costs = blas.append(costs, (cost / batch_count))
			accs = blas.append(accs, (acc*100 / batch_count))
			times = blas.append(times, (cycle_end - cycle_start))
			if print_cycle:
				print(f'Duration: {f"{blas.around(times[-1], 2)}s":<5}', f'/ Accuracy: {blas.around(accs[-1], 2)}%   ')
		train_end = time.time()
		train_time = blas.around(train_end - train_start, 2)
		train_mins = int((train_time) // 60)
		train_secs = int((train_time) - (train_mins * 60))
		avg_time = blas.around(blas.average(times), 2)
		times = blas.around(times, 2)
		accs = blas.around(accs, 2)
		costs = blas.around(costs, 4)
		if MODEL_PRINTING:
			print(f'\n TOTAL TRAINING DURATION:\t {train_mins}m : {train_secs}s')
			print(f' AVG. TRAINING CYCLE DURATION:\t {avg_time}s')
		return costs, accs, times, train_time, avg_time

	def test(self, test_x, test_y):
		if MODEL_PRINTING: print(f'\n TESTING...')
		output = self.forward(test_x)
		acc = blas.around(100 * blas.count_nonzero(blas.argmax(output, axis=1) == blas.argmax(test_y, axis=1)) / test_x.shape[0], 2)
		if MODEL_PRINTING: print(f'   TEST ACCURACY: {acc}%')
		return acc, output

class Layer:
	def __init__(self):
		self.input = None
		self.output = None
		self.type = 'BASE LAYER CLASS'

	def forward(self, input):
		pass

	def backward(self, grad):
		pass

class Dense(Layer):
	def __init__(self, weights=[10,10]):
		self.type = 'Dense'
		self.weights = self.random_init('He', weights[0], weights[1]) if isinstance(weights, list) else weights.copy()

	def forward(self, input):
		self.input = input
		self.output = input.dot(self.weights)
		return self.output

	def backward(self, grad, lr):
		dW = blas.dot(grad.T, self.input).T
		dA = blas.dot(grad, self.weights.T)
		self.weights -= dW * lr
		return dA

	def random_init(self, rand_type, input_size, output_size):
		if rand_type == 'He':
			return blas.random.randn(input_size, output_size) * blas.sqrt(2.0/input_size) # He initialization
		else:
			return blas.random.randn(input_size, output_size) # standard initialization

class Softplus(Layer):
	def __init__(self):
		self.type = 'Softplus'

	def forward(self, z):
		self.input = z
		self.output = blas.log(1.0 + blas.exp(z))
		return self.output

	def backward(self, grad, lr):
		ez = blas.exp(self.input)
		return ez / (1.0 + ez) * grad

class Softmax(Layer):
	def __init__(self):
		self.type = 'Softmax'

	def forward(self, z):
		self.input = z
		z = z - blas.max(z, axis=1).reshape(z.shape[0], 1)
		ez = blas.exp(z)
		self.output = ez / blas.sum(ez, axis=1).reshape(z.shape[0], 1)
		return self.output

	def backward(self, grad, lr):
		return grad



if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1] == 'get_data':
		download_data()
		exit()
	# #################### DON'T CHANGE ANYTHING ABOVE THIS LINE ####################
	# ----- GREET USER & PROVIDE INFO
	greet(True)	# pass "True" to clear screen before next run; default = False

	# ----- LOAD DATA
	datasets = load_data()	# DON'T CHANGE THIS LINE

	# ----- SET PERFORMANCE RECORDING OPTIONS
	logfile = 'test-run-9'		# set path of your performance output file; default = "metrics/performance.txt"
	record_performance = True	# set "False" if you don't want to save results
	save_weights = False		# set "False" if you don't want to save weights; "weights/{logfile}-weights.txt"

	# optionally, load list of previous model weights
	weights = load_model_weights('test-run-9') if save_weights else False

	# ----- TEST DIFFERENT ARCHITECTURES & HYPERPARAMETERS
	# Define model architectures & hyperparameters.
	# - dims: list of comma-separated integers
	# - cycles: single integer
	# - lr (learning rate): single float/decimal value; ideally 0.001 - 0.01
	# - bs (batch size): single integer; larger trains faster, but less accurate; ideally 64 - 128
	# - pf (print frequency): how frequently should the model print metrics during training?
	models = [
		{'dims':[32,16], 'cycles':10, 'lr':0.001, 'bs':64, 'pf':1},
		{'dims':[64,32], 'cycles':10, 'lr':0.001, 'bs':64, 'pf':1},
	]
	# #################### DON'T CHANGE ANYTHING BELOW THIS LINE ####################


	evaluate_models(datasets, models, record_performance, save_weights, logfile)
