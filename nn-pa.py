import numpy as np
import requests as rq
import matplotlib.pyplot as plt
import time, sys, os, json
from datetime import datetime


MATH_ENV, blas = 'numpy', np
try:
	import cupy as cu
	MATH_ENV, blas = 'cupy', cu
except Exception as e:
	print(f" CuPy not found, running neural network on CPU.\n To install CuPy, visit:\n  https://docs.cupy.dev/en/stable/install.html")


# ----- SEED RNG
blas.random.seed(4)


# ----- FORMATTING UTILS
def greet(clear=False):
	if clear: os.system('cls' if os.name=='nt' else 'cls')
	print((
		"\n CuPy GPU-Powered MNIST Neural Network \n"
		" ------------------------------------- \n"
		"  This neural network uses your GPU to train on the MNIST \n"
		" dataset and learn to recognize images of hand-written digits. \n\n"
		"  If you don't have a GPU or don't have CuPy installed on \n"
		" your system, the program will use NumPy instead. The training \n"
		" will run much slower, but it will achieve the same results. \n\n"
		" Email john@discefasciendo.com with questions.\n\n Enjoy! \n\n\n"
	))
	env = 'CuPy Array Library for GPU-accelerated Computing' if MATH_ENV == 'cupy' else 'NumPy package for scientific computing'
	print(f' Utilizing {env} \n')


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
	have_train = os.path.exists('data/mnist_train.csv')
	have_test = os.path.exists('data/mnist_test.csv')
	if have_train and have_test:
		print(f'\n Loading training & testing datasets...')
		files = ['mnist_train', 'mnist_test']
		out = []
		for file in files:
			sys.stdout.write(f'  - {file}')
			load_start = time.time()
			data = np.loadtxt(f'data/{file}.csv', delimiter = ',')
			x = normalize(data[:,1:])
			y = one_hot(data[:,:1], 10)
			if MATH_ENV == 'cupy':
				x, y = cu.array(x), cu.array(y)
			load_end = time.time()
			out.append((x, y))
			print(f' ({round(load_end - load_start, 2)}s)')
		print('')
		datasets = [out[0][0], out[0][1], out[1][0], out[1][1]]
		return datasets, 'Datasets loaded...'
	else:
		return None, 'Datasets not downloaded. Download with option #1...'

def batch_data(X, Y, batch_size, cycles):
	sys.stdout.write(f' Batching training dataset... ')
	batching_start = time.time()
	train_batches = []
	for e in range(cycles):
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
	print(f'({blas.around(batching_end - batching_start, 2)}s)    ')
	return train_batches

def download_data(datasets):
	have_train = os.path.exists('data/mnist_train.csv')
	have_test = os.path.exists('data/mnist_test.csv')
	if have_train and have_test:
		return datasets, 'MNIST data already downloaded.'
	else:
		print(' Downloading MNIST data...')
		train_url = 'https://pjreddie.com/media/files/mnist_train.csv'
		test_url = 'https://pjreddie.com/media/files/mnist_test.csv'
		for url in [train_url, test_url]:
			f = url.split('/')[-1]
			sys.stdout.write(f'  - {f}')
			start = time.time()
			req = rq.get(url)
			res = req.text
			fp = open(f'data/{f}', 'w')
			fp.write(res)
			fp.close
			end = time.time()
			sys.stdout.write(f' ({blas.around(end-start, 2)}s)\n')
		return load_data()

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
	plt.suptitle(f' Best MNIST Model Predictions\n (Model: {model_idx} - Acc. {model_acc}%)', fontsize=16, fontweight='bold')
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
		fp.write(f'env;dims;cycles;learning rate;batch size;total train time;avg cycle time;training accuracy;test accuracy;training accuracy delta\n')
		fp.close()
	fp = open(f'metrics/{fn}.txt', 'a')
	for line in metrics:
		fp.write(f'{MATH_ENV};[{",".join(f"{line[0]}".split("[")[1].split("]")[0].split(", "))}];{line[1]};{line[2]};{line[3]};{line[4]};{line[5]};{line[6]};{line[7]};{line[8]}\n')
	fp.close()

def save_model_weights(model, metrics, fn='model-weights'):
	print(f' Saving best model weights to file: weights/{fn}.txt')
	fp = open(f'weights/{fn}.txt', 'w')
	fp.write(f'env;dims;cycles;learning rate;batch size;total train time;avg cycle time;training accuracy;test accuracy;training accuracy delta\n')
	fp.write(f'{MATH_ENV};[{",".join(f"{metrics[0]}".split("[")[1].split("]")[0].split(", "))}];{metrics[1]};{metrics[2]};{metrics[3]};{metrics[4]};{metrics[5]};{metrics[6]};{metrics[7]};{metrics[8]}\n')
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
	print(f'\n Loading previous model weights for: {filename}')
	parse_dims = lambda str: np.array(str[:-2].split('(')[1].split(', '), dtype=int)
	parse_weights = lambda str: np.array(str[:-1].split(','), dtype=np.float64)

	fp = open(f'weights/{filename}', 'r')
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
		print(f'\n\n TESTING MODEL {i+1}...')

		# get model hyperparameters
		dims = model['dims']
		cycles = model['cycles']
		lr = model['lr']
		pf = model['pf']

		# train & test model
		m = Net(train_x, train_y, test_x, test_y, dims, cycles, lr, pf)
		costs, accs, times, train_time, avg_time, train_acc_delta = m.train()
		test_acc, predictions = m.test()

		# save model parameters & performance
		if not isinstance(dims[0], int): dims = [d.shape[0] for d in dims[1:]]
		metrics.append([dims, costs.shape[0], lr, 64, train_time, avg_time, accs[-1], test_acc, train_acc_delta])
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
		print('')

	return best_idx+1, best_model, best_preds, best_acc, metrics, performance_data

def evaluate_models(datasets, models):
	if datasets == None:
		return 'Datasets not downloaded. Download with option #1...'
	if len(models) < 1:
		return 'No model architectures defined. Create with option #2...'

	dt = datetime.now().strftime("%m/%d/%Y_%H:%M").replace("/","-").replace(":","")
	logfile = f'{dt}'
	record_performance = True if input('\n Record performance metrics? (type "y" or "n") ') == 'y' else False
	save_weights = True if input(' Save best model weights? (type "y" or "n") ') == 'y' else False
	if record_performance or save_weights:
		lf_msg = (
			"\n Type the name of the output files ('.txt' extension will be added),"
			"\n or press enter to accept the default name. This will be used for"
			"\n metrics and/or weights logfiles."
			f"\n\n Default name(s): '{logfile}-metrics.txt' and/or '{logfile}-weights.txt'"
		)
		usr_name = input(f'{lf_msg}\n >>> ')
		logfile = logfile if usr_name == '' else usr_name

	display_models(models, False)

	# test models & get metrics/performance data
	best_idx, best_model, best_preds, best_acc, metrics, performance_data = train_models(datasets, models)

	# show predictions of best model
	show_predictions(datasets[2], best_preds, f'#{best_idx}', best_acc)

	# plot model performance & write to file
	plot_lines(performance_data, best_idx, best_acc)

	# optionally, save performance metrics of all models tested
	mlf = f'{logfile}-metrics' if f'{dt}' == logfile else logfile
	if record_performance: record_performances(metrics, mlf)

	# optionally, save weights of the best performing model
	wlf = f'{logfile}-weights' if f'{dt}' == logfile else logfile
	if save_weights: save_model_weights(best_model, metrics[best_idx-1], wlf)

	input('\n Press enter to continue...')
	return 'Performance metrics recorded.' if record_performance or save_weights else 'Models evaluated.'

def test_models(datasets):
	train_x, train_y, test_x, test_y = datasets
	weights = os.listdir('weights')
	weights.remove('README.txt')
	if len(weights) == 0:
		return 'No weights to load.'
	else:
		wts = [(f'{i+1}: {w[:-4]}') for i,w in enumerate(weights)]
		print(f'\n Available weights: {wts}\n')
		try:
			idx = int(input(' Type the number of the weight file you want to test.\n '))
			if 1 <= idx <= (len(weights)+1):
				dims = load_model_weights(weights[idx-1])
				m = Net(train_x, train_y, test_x, test_y, dims, 1, 0.005, 10)
				test_acc, predictions = m.test()
				show_predictions(test_x, predictions, weights[idx-1], test_acc)
				return f'Test Accuracy: {test_acc}%'
			else:
				return 'Invalid weight selection. Please try again.'
		except Exception as e:
			return f'Invalid weight selection. Please try again.\n >>> {e}'

# ----- USER UTILS
def define_model_architectures(models):
	weights = os.listdir('weights')
	weights.remove('README.txt')
	updated_models = []

	done, first = False, True
	while not done:
		if not first: print('\n')
		if first: first = False
		dims = []
		dims_input = '\n Enter the layer dimensions for the model as a comma-separated list. (e.g.: "256,64,32")\n >>> '
		if len(weights) > 0:
			prntd = [(f'{i+1}: {f[:-4]}') for i,f in enumerate(weights)]
			print(f'\n Available weights: {prntd}\n')
			load_idx = input(' Load model weights? Type file number, or press enter to skip\n ')
			if not load_idx == "":
				idx = int(load_idx)
				if weights[idx-1] in weights:
					dims = load_model_weights(weights[idx-1])
				else:
					print(f' "#{idx-1}" not in available weights. Try again... ')
					time.sleep(1)
			else: dims = input(dims_input).split(',')
		else: dims = input(f'\n No weights to load. {dims_input[2:]}').split(',')

		valid = True
		for i,d in enumerate(dims):
			t = f'{type(d)}'
			if not 'ndarray' in t:
				try: dims[i] = int(d)
				except Exception as e:
					valid = False

		if not valid:
			print(f' Invalid weight entry. Try again... ')
			time.sleep(1)
			continue

		cycles = int(input('\n How many cycles should the model train for?\n >>> '))
		lr = float(input('\n What is the learning rate? (Ideally between 0.001 - 0.01)\n >>> '))
		pf = round(cycles * 0.05/10)*10
		pf = 1 if pf == 0 else pf
		updated_models.append({ 'dims': dims, 'cycles': cycles, 'lr': lr, 'pf': pf })

		add = True if input('\n Add another model? (type "y" or "n") ') == 'y' else False
		if not add: done = True

	return updated_models, 'Model architectures updated...'

def display_help_menu(msg=''):
	print(f' Options:')
	actions = [
		{'title': 'Download Data', 'desc': 'Download MNIST datasets.'},
		{'title': 'Define Model Architectures', 'desc': 'Set model architecture options.'},
		{'title': 'View Model Architectures', 'desc': 'View defined architectures.'},
		{'title': 'Evaluate Models', 'desc': 'Evaluate defined models.'},
		{'title': 'Test Weights', 'desc': 'Test pre-trained weights.'},
		{'title': 'Quit', 'desc': 'Quit the program.'},
	]
	for i,action in enumerate(actions):
		print(f'  {i+1}: {action["title"]} - {action["desc"]}')

	if not msg == '': print(f'\n\n >>> {msg}\n')

def display_models(models, pause=True):
	if len(models) < 1:
		return 'No models specified. Define a model with option #2.'
	else:
		print(f'\n Current model architectures.')
		for i,m in enumerate(models):
			m['dims'] = m['dims'] if type(m['dims'][0]) == int else [d.shape[0] for d in m['dims'][1:]]
			print(f' - Model {i+1}: {m}'.replace("'","").replace('lr','learning rate').replace('pf','print frequency'))
		if pause: input('\n Press enter to continue.')
		return ''

# ----- NEURAL NETWORK CLASSES
class Net:
	def __init__(self, X, Y, x, y, layers=[256,128], cycles=3, lr=0.001, print_freq=10, batch_size=64):
		print(f'\n Initializing network... (cycles={cycles}, learning rate={lr})')
		self.train_x = X
		self.train_y = Y
		self.test_x = x
		self.test_y = y
		self.layers = self.init_layers(layers)
		self.cycles = cycles
		self.lr = lr
		self.print_freq = print_freq
		self.batch_size = batch_size

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
			n = self.train_x.shape[1]
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

	def train(self):
		print(f'\n TRAINING...')
		m, n = self.train_x.shape
		costs, accs, times = blas.array([]), blas.array([]), blas.array([])

		# Batch training data preemptively to speed up the etraining process.
		batches = batch_data(self.train_x, self.train_y, self.batch_size, self.cycles)
		batch_count = len(batches[0])
		batch_size = batches[0][0][0].shape[0]

		test_acc_tmp = 0.0

		train_start = time.time()
		for cycle in range(self.cycles):
			if (cycle > 0) and (accs[-1] >= 100.0):
				break
			# if (cycle > 0) and (test_acc_tmp == 0.0) and (accs[-1] >= 100.0):
			# 	test_acc_tmp, _ = self.test()
			# 	print(f' INIT TEST ACC @ 100% TRN ACC: {test_acc_tmp}')
			# if (cycle == self.cycles-1) and (not (test_acc_tmp == 0.0)):
			# 	delta = test_acc_tmp
			# 	test_acc_tmp, _ = self.test()
			# 	delta = test_acc_tmp - delta
			# 	print(f' FINAL TEST ACC @ 100% TRN ACC: {test_acc_tmp} (delta: {delta})')
			current_batches = batches[cycle]
			cycle_start = time.time()
			cost, acc = 0, 0
			print_cycle = True if MODEL_PRINTING and ((cycle==0) or ((cycle+1) % self.print_freq == 0) or (cycle == self.cycles-1)) else False

			if print_cycle: sys.stdout.write(f' {f" {cycle+1}/{self.cycles} >> ":>12}')

			for b,batch in enumerate(current_batches):
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
				print(f'{f"Duration: {blas.around(times[-1], 2)}s":<15} / {f"Accuracy: {blas.around(accs[-1], 5)}%"}')

		train_end = time.time()
		train_time = blas.around(train_end - train_start, 2)
		train_mins = int((train_time) // 60)
		train_secs = int((train_time) - (train_mins * 60))
		avg_time = blas.around(blas.average(times), 2)
		times = blas.around(times, 2)
		accs = blas.around(accs, 5)
		costs = blas.around(costs, 5)
		train_acc_delta = blas.around(accs[-1] - accs[0], 2)
		if MODEL_PRINTING:
			print(f'\n TOTAL TRAINING DURATION:\t {train_mins}m : {train_secs}s')
			print(f' AVG. TRAINING CYCLE DURATION:\t {avg_time}s')
		return costs, accs, times, train_time, avg_time, train_acc_delta

	def test(self):
		print(f'\n TESTING...')
		x, y = self.test_x, self.test_y
		output = self.forward(x)
		acc = blas.around(100 * blas.count_nonzero(blas.argmax(output, axis=1) == blas.argmax(y, axis=1)) / x.shape[0], 5)
		print(f'   TEST ACCURACY: {acc}%')
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
			return blas.random.randn(input_size, output_size)

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
	running = True	# SET TO "False" TO QUIT
	models = []		# DEFINE ARCHITECTURES & HYPERPARAMETERS

	# GREET USER & PROVIDE INFO
	greet(True)

	# BEGIN RUNTIME
	datasets, usr_msg = load_data()
	while running:
		greet(True)

		# DISPLAY PROGRAM OPTIONS & CLEAR PREVIOUS USER MSG
		display_help_menu(usr_msg)
		usr_msg = ''

		# PROMPT USER FOR CHOICE
		choice = input(f'\n What would you like to do? (Type 1-5 and press enter.)\n ')
		try:
			choice = int(choice)
			if choice in [1,2,3,4,5]:
				clear_screen()
				if choice == 1: datasets, usr_msg = download_data(datasets)
				if choice == 2: models, usr_msg = define_model_architectures(models)
				if choice == 3: usr_msg = display_models(models)
				if choice == 4: usr_msg = evaluate_models(datasets, models)
				if choice == 5: usr_msg = test_models(datasets)
			elif choice == 6:
				print('\n Quitting...')
				running = False
			else:
				usr_msg = 'Invalid option. Please select a number 1-5 and press enter...'
		except Exception as e:
			print(f' ERROR >>> {e}')
			time.sleep(5)
			usr_msg = f'Unable to understand selection. Please type a number 1-5 and press enter...\n {e}'
