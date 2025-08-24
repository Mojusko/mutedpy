import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import string
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


class EarlyStopping():
	"""
	Early stopping to stop the training when the loss does not improve after
	certain epochs.
	"""

	def __init__(self, patience=5, min_delta=0):
		"""
		:param patience: how many epochs to wait before stopping when loss is
			   not improving
		:param min_delta: minimum difference between new loss and old loss for
			   new loss to be considered as an improvement
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False

	def __call__(self, val_loss):
		if self.best_loss == None:
			self.best_loss = val_loss
		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
			if self.counter >= self.patience:
				print('INFO: Early stopping')
				self.early_stop = True


class neural_net_model(nn.Module):

	def __init__(self, batchsize=1000, validation_set_ratio=0.0, X_features = None):
		super().__init__()

		self.device = 'cpu'
		self.nr_epochs = 500
		self.batch_size = batchsize

		# define model
		self.fc1 = nn.Linear(5*36,64)#.to(self.device)
		self.bn1 = nn.BatchNorm2d(64)#.to(self.device)
		self.fc2 = nn.Linear(64, 1)#.to(self.device)
		self.fc3 = nn.Linear(1, 1)#.to(self.device)

		# embeddings
		mapping = {}
		self.one_hot_mapping = {}
		for i, s in enumerate(string.ascii_uppercase + string.digits):
			mapping[s] = i
			aux = np.zeros(36).astype(float)
			aux[i] = 1
			self.one_hot_mapping[s] = aux

		# generate validation set
		self.validation_set_ratio = validation_set_ratio


	def forward(self,x,z = None):
		x = self.fc1(x)
		x = self.bn1(x)
		x = torch.nn.functional.relu(x)
		x = self.fc2(x)
		if z is not None:
			x = x + self.fc3(z)
		return x

	def weight_reset(self, m):
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			m.reset_parameters()

	def create_batch_t_from_inds(self, X_seq, inds, y=None, X_fea=None):
		x_batch = []
		for sec in X_seq[inds]:
			if len(sec[1:-1]) < 3:
				mid = (3 - len(sec[1:-1])) * "0" + sec[1:-1]
				sec = sec[0] + mid + sec[-1]
			elem = np.array([self.one_hot_mapping[s] for s in sec]).flatten().astype(float)
			x_batch.append(elem)
		x_batch = torch.from_numpy(np.array(x_batch)).to(self.device).float()

		if y is not None:
			y_batch = torch.from_numpy(y[inds]).to(self.device).float()
			return x_batch, y_batch
		else:
			return x_batch

	def generate_validation_set(self, X, y):
		ntrain = int((1 - self.validation_set_ratio) * X.shape[0])
		nval = X.shape[0] - ntrain

		ind = np.random.choice(X.shape[0], X.shape[0], replace=False)

		Xtrain = X[ind[0:ntrain]]
		ytrain = y[ind[0:ntrain]]
		Xval = X[ind[ntrain:]]
		yval = y[ind[ntrain:]]

		return Xtrain, ytrain, Xval, yval

	def fit(self, X, y, X_fea=None, verbose=True):

		Xtrain, ytrain, Xval, yval = self.generate_validation_set(X, y)
		n_train = Xtrain.shape[0]

		self.criterion = nn.SmoothL1Loss()
		optimizer = torch.optim.Adam(self.parameters(), lr=5 * 1e-4)

		self.apply(self.weight_reset)
		val_losses = []

		early_stopping = EarlyStopping(patience=10)

		for e in range(self.nr_epochs):
			self.train()
			for it in range(n_train // self.batch_size + 1):
				batch_inds = np.random.choice(n_train, self.batch_size, replace=False)
				x_batch, y_batch = self.create_batch_t_from_inds(Xtrain, batch_inds, y=ytrain)

				output = self.forward(x_batch)
				loss = self.criterion(output, y_batch)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			val_epoch_loss = self.eval(Xval, yval)
			early_stopping(val_epoch_loss)
			if early_stopping.early_stop:
				break

			if (e % 2 == 0) and (verbose == True):
				# print (self.eval(Xval, yval),"epoch {} train".format(e))
				print("epoch %d train_loss: %4.4f val_loss:%4.4f " % (
				e, self.eval(Xtrain, ytrain), self.eval(Xval, yval)))

	def eval(self, X_seq, y, X_features=None):
		self.eval()
		inds = np.arange(X_seq.shape[0])
		if X_features is None:
			x_batch, y_batch = self.create_batch_t_from_inds(X_seq, inds, y=y)
			err = self.criterion(self.forward(x_batch, None), y_batch)
		else:
			x_batch, y_batch = self.create_batch_t_from_inds(X_seq, inds, y=y)
			err = self.criterion(self.forward(x_batch, None), y_batch)

		return err

	def predict(self, X):
		self.model.eval()
		inds = np.arange(X.shape[0])
		x_batch = self.create_batch_t_from_inds(X, inds, y=None)
		pred = self.forward(x_batch)
		return pred


if __name__ == "__main__":
	dts = pd.read_csv('../../notebooks/docking-analysis/amie-ibo.csv')
	dts = dts.dropna()
	mask = dts['Fitness2'] != 'NS'
	dts = dts[mask]

	X = dts['Mutation'].values
	y = dts['Fitness2'].values.astype(float).reshape(-1, 1)

	NN = neural_net_model(validation_set_ratio=0.1)
	NN.fit(X, y)

	scorer = lambda model, X, y: r2_score(y, model.predict(X).detach().numpy())
	kf = KFold(n_splits=10)
	kf.get_n_splits(X)
	scores = []
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		NN.fit(X_train, y_train)
		score = scorer(NN, X_test, y_test)
		scores.append(score)
	print(scores)
	print ("In the absence of docking affinity:")
	print(np.mean(scores), np.median(scores))












