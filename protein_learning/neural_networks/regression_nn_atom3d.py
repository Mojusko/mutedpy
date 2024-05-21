import argparse
import os
import time
import datetime
import pandas as pd
from mutedpy.utils.loaders.loader_basel import BaselLoader
from mutedpy.utils.sequences.sequence_utils import create_neural_mutations

from torch_geometric.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

from mutedpy.utils.scoring.metrics import METRICS, DATASET_METRICS, EVAL_METRICS
from mutedpy.utils.data_splits import *

DATA_DIR = "../../../data/streptavidin/rosetta/"
PROCESSED_DIR = "../../../data/streptavidin/features"
LOG_DIR = "."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ProtFitnessDataset(torch.utils.data.Dataset):

	def __init__(self,
				 dataset: str = "None",
				 raw_dir: str = None,
				 processed_dir: str = None,
				 split :str = None,
				 mode: str = 'train',
				 task: str = 'regression'):

		with open(f"{split}", "r") as f:
			splits = json.load(f)

		self.task = task
		self.pdb_ids = splits[mode]
		self.pdb_files = [f"{processed_dir}/{pdb_id}.pth"
						  for pdb_id in self.pdb_ids
						  if os.path.exists(f"{processed_dir}/{pdb_id}.pth")]
		print ("Loading dataset to RAM ...")
		self.x = [torch.load(pdb_file, map_location='cpu') for pdb_file in self.pdb_files]
		print ("Dataset loaded.")
	def __len__(self):
		return len(self.pdb_files)

	def __getitem__(self, idx):
		graph = self.x[idx]
		return graph


class GNN(torch.nn.Module):

	def __init__(self, num_features: int, hidden_dim: int,
				 fc_layers: bool = False):
		super(GNN, self).__init__()
		self.conv1 = GCNConv(num_features, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
		self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
		self.conv4 = GCNConv(hidden_dim * 4, hidden_dim * 4)
		self.conv5 = GCNConv(hidden_dim * 4, hidden_dim * 8)
		self.fc_layers = fc_layers
		if fc_layers:
			self.fc1 = nn.Linear(hidden_dim * 8, hidden_dim * 4)
			self.fc2 = nn.Linear(hidden_dim * 4, 1)

	def forward(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv3(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv4(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv5(x, edge_index, edge_weight)
		x = global_add_pool(x, batch)
		x = F.relu(x)
		if self.fc_layers:
			x = F.relu(self.fc1(x))
			x = F.dropout(x, p=0.25, training=self.training)
			return self.fc2(x).view(-1)
		return x


def train_loop(model: nn.Module, loader: DataLoader, optimizer,
			   task: str = 'regression', pos_weight: float = 10.0):
	model.train()

	loss_all = 0
	total = 0
	for idx, data in enumerate(loader):

		data = data.to(DEVICE)
		optimizer.zero_grad()
		output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
		loss = F.mse_loss(output.float(), data.y.float())

		loss.backward()
		loss_all += loss.item() * data.num_graphs
		total += data.num_graphs
		optimizer.step()

		if idx % 100 == 0:
			avg_loss = np.round(loss_all / total, 4)
			avg_loss = np.round(np.sqrt(avg_loss), 4)
			print(f'After {idx} steps: Loss: {avg_loss}', flush=True)

	return np.sqrt(loss_all / total)


@torch.no_grad()
def test(model, loader, metrics, task: str = 'regression', pos_weight: float = 10.0):
	model.eval()

	loss_all = 0
	total = 0

	y_true = []
	y_pred = []

	for data in loader:
		data = data.to(DEVICE)
		output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
		loss = F.mse_loss(output, data.y)
		loss_all += loss.item() * data.num_graphs
		total += data.num_graphs
		y_true.extend(data.y.tolist())
		y_pred.extend(output.tolist())

	metric_dict = {}
	metric_dict['loss'] = np.sqrt(loss_all / total)

	for metric in metrics:
		metric_fn = METRICS.get(metric)[0]
		metric_dict[metric] = metric_fn(y_true, y_pred)

	return metric_dict


def train(args, log_dir: str, test_mode: bool = False):


	train_dataset = ProtFitnessDataset(dataset=args.dataset, raw_dir=args.data_dir,
									   processed_dir=args.processed_dir,
									   split=args.split, mode='train', task=args.task)

	val_dataset = ProtFitnessDataset(dataset=args.dataset, raw_dir=args.data_dir,
									 processed_dir=args.processed_dir,
									 split=args.split, mode='val', task=args.task)

	if test_mode:
		test_dataset = ProtFitnessDataset(dataset=args.dataset, raw_dir=args.data_dir,
										  processed_dir=args.processed_dir,
										  split=args.split, mode='test', task=args.task)

	train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
	val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)

	for data in train_loader:
		num_features = data.num_features
		break

	metric_names = DATASET_METRICS
	eval_metric = EVAL_METRICS

	metrics = {}

	for metric_name in metric_names:
		metrics[metric_name], _, _ = METRICS.get(metric_name)

	_, best_eval_metric, compare_fn = METRICS.get(eval_metric)

	model = GNN(num_features, hidden_dim=args.hidden_dim, fc_layers=True).to(DEVICE)

	model.to(DEVICE)

	# best_val_loss = 999
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

	for epoch in range(1, args.num_epochs + 1):
		start = time.time()
		train_loss = train_loop(model, train_loader, optimizer, task=args.task, pos_weight=args.pos_weight)
		val_metrics = test(model, val_loader, metrics, task=args.task, pos_weight=args.pos_weight)
		improvement = compare_fn(val_metrics[eval_metric], best_eval_metric)
		if improvement:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': train_loss,
				'val_metrics': val_metrics
			}, os.path.join(log_dir, f'best_weights.pt'))
			# plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}.png'))
			best_eval_metric = val_metrics[eval_metric]
		elapsed = (time.time() - start)
		print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
		print_msg = f"\tTrain Loss: {np.round(train_loss, 4)}"
		for metric, metric_val in val_metrics.items():
			print_msg += f", Val {metric}: {np.round(metric_val, 4)}"
		print(print_msg, flush=True)

	return eval_metric, best_eval_metric


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default=DATA_DIR)
	parser.add_argument("--processed_dir", type=str, default=PROCESSED_DIR)
	parser.add_argument("--task", type=str, default="regression")
	parser.add_argument('--log_dir', type=str, default=LOG_DIR)
	parser.add_argument("--dataset", type=str, default="streptavidin")
	parser.add_argument("--split", type=str, default='random')
	parser.add_argument("--pos_weight", type=float, default=10.0)
	parser.add_argument('--batch_size', type=int, default=5)
	parser.add_argument('--hidden_dim', type=int, default=128)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--learning_rate', type=float, default=1e-6)
	parser.add_argument('--exp_name', default=None)
	args = parser.parse_args()

	log_dir = args.log_dir
	torch.set_default_dtype(torch.float32)

	exp_name = args.exp_name

	filename = "../../../data/streptavidin/5sites.xls"
	loader = BaselLoader(filename)
	dts = loader.load()

	filename = "../../../data/streptavidin/2sites.xls"
	loader = BaselLoader(filename)
	total_dts = loader.load(parent='SK', positions=[112, 121])
	total_dts = loader.add_mutations('T111T+N118N+A119A', total_dts)

	total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)
	total_dts = create_neural_mutations(total_dts)
	total_dts['LogFitness'] = np.log10(total_dts['Fitness'])

	split_name = "random_split.json"
	names = total_dts["Mutation"]
	split_data_names_save_to_json(split_name,names)

	args.split = split_name


	if exp_name is None:
		exp_name = f"Atom3D"
		now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		exp_name += f"_{now}"

	log_dir = os.path.join(log_dir, exp_name)

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	train(args, log_dir)
