import pickle
import mutedpy
from mutedpy.protein_learning.regression.regression_basis import ProteinKernelLearner
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import esm.pretrained
import torch.optim as optim
import numpy as np
import pandas as pd
from mutedpy.protein_learning.neural_networks.neural_nets import NeuralNetwork
from mutedpy.utils.sequences.sequence_utils import from_integer_to_variants

class SequenceDataset(Dataset):
    def __init__(self, seqs, y,  Embedding, classes = None, type = 'pickle', reshape = False):
        self.seqs = seqs
        self.y = y
        self.Embedding = Embedding
        self.classes = classes
        self.type = type
        self.reshape = reshape

    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):

        if self.type == 'mongo':

            self.Embedding.connect()
            self.db = self.Embedding.get_db()
            res = self.db.find_one({"params": self.seqs[idx]})
            emb = torch.Tensor(res['embedding-mean']).double().view(-1)
            y = self.y[idx,:]
            self.Embedding.close()

        elif self.type == 'pickle':
            if self.reshape:
                emb = torch.Tensor(self.Embedding.embed_seq([self.seqs[idx]])).double().view(-1)
            else:
                emb = torch.Tensor(self.Embedding.embed_seq([self.seqs[idx]])).double()
            y = self.y[idx,:]

        elif self.type == 'esm':
            emb = self.Embedding.embed_seq([self.seqs[idx]])[0,:,:]
            y = self.y[idx, :]

        else:
            raise NotImplementedError("Error.")
        return emb, y

class ProteinKernelLearnerBachedClassification(ProteinKernelLearner):

    def __init__(self, classes = 3, loss = 'logit',
                 emb_size = 1280,
                 datatype = "pickle",
                 arch = "linear",
                 seq_len = 99,
                 num_heads = 4,
                 learning_rate= 1e-2,
                 epochs = 1,
                 batch_size = 100,
                 **kwargs):

        super().__init__(**kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.classes = classes
        self.loss = loss
        self.num_heads = num_heads
        self.datatype = datatype
        self.emb_size = emb_size
        self.mid_layer = 512
        self.end_layer = 128
        self.arch = arch
        self.learning_rate = learning_rate
        self.seq_len = seq_len

    def load(self):
        self.Embedding = self.feature_loader

    def get_loss(self, emb, y, model, device, lam = 1e-10):

        if self.loss == 'logit' or self.loss == 'logit_nn':
            logits = torch.sigmoid(model(emb))
            loss = - torch.mean(y* torch.log(logits) + (
                        1 - y) * torch.log(1 - logits))

        elif self.loss == 'squared':
            sigma = float(1./0.1/np.sqrt(2))
            mu = model(emb)
            y_value = y*torch.Tensor([1.2,1.7,3.0]).to(device)
            loss = -torch.sum(torch.log(0.5 * (1 + torch.erf((y_value - mu) * sigma ))))

            #loss = -torch.mean(torch.log(torch.erf(torch.abs(y_value - mu)/sigma**2)))
        elif self.loss == 'mse':

            y_value = torch.max(y*torch.Tensor([1.2,1.7,3.0]).to(device), dim = 1)[0]
            loss = torch.mean((model(emb) - y_value)**2)

        elif self.loss == "laplace":
            mu = model(emb)
            y_value = torch.max(y * torch.Tensor([1.2, 1.7, 3.0]).to(device), dim=1)[0]
            b = 0.1
            loss = -torch.sum(torch.log(0.5 + 0.5 * torch.sign(y_value - mu) * (1 - torch.exp(-torch.abs(y_value - mu) / b))))

        if "weighted" in self.arch:
            # TODO: impement weight decay on the weighted architecture
            pass
        else:
            linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
            linear_losses = sum([torch.sum(model.weight**2) for model in linear_layers])
            loss = loss + lam * linear_losses
        return loss

    def predict_values(self, emb, model):
        if self.loss == 'logit' or self.loss == "logit_nn":
            logits = torch.sigmoid(model(emb))
            pred = (logits > 0.5).double().sum(dim=1)

        elif self.loss == 'squared' or self.loss == "mse" or  self.loss == "laplace":
            mu = model(emb).view(-1)
            logits = torch.vstack([mu > 1.2, mu> 1.7, mu > 3.0]).double().T
            pred = (logits > 0.5).double().sum(dim=1)

        return pred

    def predict(self):

        self.model.eval()
        dataset_test = SequenceDataset(self.seq_test, self.y_test, self.Embedding, type=self.datatype)
        data_loader_test = DataLoader(dataset_test, batch_size=1000, shuffle=False, num_workers=1)
        predictions = []
        for idy, batch_data_val in enumerate(data_loader_test):
            batch_emb_val, batch_y_val = batch_data_val
            pred = self.predict_values(batch_emb_val, self.model)
            predictions.append(pred)
        self.pred = torch.cat(predictions)
        return self.pred

    def split_to_validation(self, split, master_seq = None, master_y = None):

        if master_seq is None or master_y is None:
            master_seq = self.seq_train
            master_y = self.y_train
        else:
            pass

        # shuffle the data
        N = len(master_seq)
        perm = torch.randperm(N)
        seq = [master_seq[i] for i in perm]
        y = master_y[perm, :].int()

        # split to train and validation
        train_size = int(split * len(self.seq_train))
        seq_train = seq[:train_size]
        seq_val = seq[train_size:]

        y_train = y[:train_size,:]
        y_val = y[train_size:,:]

        return y_train, y_val, seq_train, seq_val

    def init(self, device = 'cpu'):
        self.load()

        if self.loss == 'logit':
            output_size = self.classes

        elif self.loss == "squared" or self.loss == "mse" or self.loss == "laplace":
            output_size = 1

        else:
            raise NotImplementedError("Not implemented.")

        NN = NeuralNetwork(self.arch,
                                   self.emb_size,
                                   self.mid_layer,
                                   self.end_layer,
                                   output_size=output_size,
                                   seq_len = self.seq_len,
                                    num_heads =self.num_heads)
        self.model = NN.get_model()

        self.model.double()
        self.model.to(device)

        self.hook_output = {}

        if "weighted" in self.arch:
            pass
            # TODO: implement getting of last layer

        else:
            def hook(module, input, output):
                self.hook_output['last_layer'] = output

            self.model[-2].register_forward_hook(hook)

    def fit(self,
            learning_rate = 1e-2,
            device = "cpu",
            test_split = False):

        epochs = self.epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        self.init(device = device)

        # initialize optimizer
        # split to train and validation

        y_train, y_val, seq_train, seq_val = self.split_to_validation(split=0.9)

        if test_split:
            y_test, y_val, seq_test, seq_val = self.split_to_validation(split=0.5, master_seq=seq_val, master_y=y_val)

        #optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        #print (list(self.model.parameters()))
        print ("Architecture summary:")
        for name, param in self.model.named_parameters():
            print(name, param.shape)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print ("COUNT SUMMARY")
        print (total_params)
        print (trainable_params)

        N_train = len(seq_train)
        N_val = len(seq_val)
        print ("Number of training points", N_train)
        print("Number of validation points", N_val)
        print("Embeddings:",  self.Embedding.get_m_list())

        dataset = SequenceDataset(seq_train, y_train, self.Embedding, type = self.datatype)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 16)

        print ("Start training")
        # optimization loop for the model
        for e in range(epochs):

            for idx, batch_data in enumerate(data_loader):
                #print ("iter:",idx, end = ', ')
                self.model.train()

                batch_emb, batch_y = batch_data
                #print(batch_emb.size())

                if len(batch_emb.size())<=2:
                    batch_emb = batch_emb.unsqueeze(2)

                loss = self.get_loss(batch_emb.to(device), batch_y.to(device), self.model, device)

                # make optimization step
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                #print (idx, loss)

                # validation
                if idx % 100 == 0:
                    self.model.eval()

                    dataset_val = SequenceDataset(seq_val, y_val, self.Embedding, type=self.datatype)
                    data_loader_val = DataLoader(dataset_val, batch_size=100, shuffle=False, num_workers=16)
                    validation_loss = []
                    predictions = []

                    for idy, batch_data_val in enumerate(data_loader_val):
                        batch_emb_val, batch_y_val = batch_data_val
                        validation_loss.append(self.get_loss(batch_emb_val.to(device), batch_y_val.to(device), self.model,device).detach().to("cpu"))
                        predictions.append(self.predict_values(batch_emb_val.to(device), self.model).detach().to("cpu"))
                        del batch_data_val, batch_y_val

                    validation_loss2 = torch.mean(torch.stack(validation_loss))
                    predictions2 = torch.cat(predictions)

                    truth = (y_val > 0.5).float().sum(dim=1)
                    accuracy2 = (predictions2 == truth).float().mean()


                    print("epoch:", e, "iter:", idx, "\nloss:", loss, "\nvalidation loss:", validation_loss2,"\naccuracy:", accuracy2)

                    for i in range(0,self.classes+1):
                        mask = (i == truth)
                        base = (truth[mask] == predictions2[mask])
                        print("base for "+str(i)+":",mask.float().mean(), "accuracy:", base.float().mean())

            if test_split:
                self.model.eval()
                dataset_test = SequenceDataset(seq_test, y_test, self.Embedding, type=self.datatype)
                data_loader_test = DataLoader(dataset_test, batch_size=1000, shuffle=False, num_workers=1)
                test_loss = []
                predictions = []

                for idy, batch_data_val in enumerate(data_loader_test):
                    batch_emb_val, batch_y_val = batch_data_val
                    test_loss.append(
                        self.get_loss(batch_emb_val.to(device), batch_y_val.to(device), self.model, device).to("cpu"))
                    predictions.append(self.predict_values(batch_emb_val.to(device), self.model).to("cpu"))

                test_loss = torch.mean(torch.stack(test_loss))
                predictions = torch.cat(predictions)

                truth = (y_test > 0.5).float().sum(dim=1)
                accuracy = (predictions == truth).float().mean()

                print("test loss:", test_loss, "accuracy:",accuracy)

                for i in range(0, self.classes+1):
                    mask = (i == truth)
                    base = (truth[mask] == predictions[mask])
                    print("base for " + str(i) + ":", mask.float().mean(), "accuracy:", base.float().mean())
        self.fitted = True
    def last_layer_embed(self, seq, num_worker = 16):
        dataset = SequenceDataset(seq, torch.zeros(size = (len(seq),1)), self.Embedding, type = self.datatype )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers = num_worker)

        last_layer = []
        self.model.eval()
        y_out = []
        for idx, batch_data in enumerate(data_loader):
            batch_emb, batch_y = batch_data
            #print (batch_emb.size())
            y_new = self.model(batch_emb)
            last_layer.append(self.hook_output['last_layer'])
            y_out.append(y_new)
        last_layer = torch.vstack(last_layer)
        y_out = torch.vstack(y_out)
        return last_layer, y_out

    def save_model(self, name = '', save_loc ="./"):

        self.model.to("cpu")
        dataset = SequenceDataset(self.seq, self.y, self.Embedding, type = self.datatype )
        data_loader = DataLoader(dataset, batch_size=10000, shuffle=False, num_workers = 16)
        last_layer = []
        self.model.eval()

        for idx, batch_data in enumerate(data_loader):
            batch_emb, batch_y = batch_data
            emb = self.model(batch_emb)
            last_layer.append(self.hook_output['last_layer'])

        torch.save(self.model.state_dict(), save_loc + "params.pt")
        last_layer = torch.vstack(last_layer)

        print ("dimensions of last-layer", last_layer.size())
        pickle.dump(last_layer,open(save_loc +name+ "_last_layer.pickl","wb"))
        pickle.dump(self.y,open(save_loc +name+ "_y.pickl","wb"))
        pickle.dump(self.seq, open(save_loc +name+ "_seq.pickl", "wb"))
        torch.save(self.model.state_dict(), save_loc + name+"_params.pickl")

    def load_model(self, load_loc ="./"):
        self.model.load_state_dict(torch.load(load_loc))
        self.model.eval()


    def evaluate_metrics_on_splits(self,
                                   split_location=None,
                                   no_splits=10,
                                   n_test=150,
                                   output_file='output.txt',
                                   special_identifier='',
                                   scores = {'acc': 3}):
        if split_location is None:
            dts = self.cv_split_eval(splits=no_splits, n_test=n_test)
        else:
            dts = self.load_splits_eval(split_location, splits=no_splits)

        splits = no_splits
        final_results = {}

        for name in scores.keys():
            val = torch.zeros(size=(scores[name], splits)).double()
            final_results[name] = val
        def evaluate(a):
            i, d = a
            return self.evaluate_on_split(i,
                                          d,
                                          splits=splits,
                                          special_identifier=special_identifier,
                                          scores=scores,
                                          save_model=False)

        results = [evaluate(a) for a in enumerate(dts)]

        print (results)
        for j,l in enumerate(results):
            for key in l.keys():
                    final_results[key][:,j] = l[key].view(-1)

        # flatten the ones with mulitple arrays - put them
        keys = final_results.keys()
        datas = []
        names = []

        for key in keys:
            if final_results[key].size()[0] > 1:
                for j in range(final_results[key].size()[0]):
                    names.append(key+"_"+str(j))
                    datas.append(final_results[key][j,:])
            else:
                datas.append(final_results[key])
                names.append(key)

        #self.save_results(names, datas, output_file=output_file)


    def evaluate_on_split(self, index, d, splits=10, special_identifier="",
                          save_model=True, scores = {'acc': 3}):
        torch.set_num_threads(self.njobs)

        print(index + 1, "/", splits)

        # test
        self.x_test = d[0]
        self.y_test = d[1]
        self.seq_test = from_integer_to_variants(d[0])

        # train
        self.x_train = d[2]
        self.y_train = d[3]
        self.seq_train = from_integer_to_variants(d[2])
        self.split = index

        self.fit()
        predictions = self.predict()

        res = {}
        res['acc'] = []

        self.model.eval()

        truth = (self.y_test > 0.5).float().sum(dim=1)
        accuracy = (predictions == truth).float()
        accuracy = accuracy.mean()
        accuracy_per_class = []


        for i in range(1, self.classes+1):
            mask = (i == truth)
            base = (truth[mask] == predictions[mask])
            print("base for " + str(i) + ":", mask.float().mean(), "accuracy:", base.float().mean())
            accuracy_per_class.append(base.float().mean())

        res['acc'] = torch.vstack(accuracy_per_class)

        self.mu = self.pred
        self.std = self.pred*0

        print ( self.y_test.size(), self.mu.size(), self.std.size())
        self.save_predictions(id=str(index), special_identifier=special_identifier, train = False, test = True)

        # if save_model:
        #     self.save_model(save_loc=self.results_folder)
        #
        # self.save_plot(id=str(index), special_identifier=special_identifier)

        return res

    def save_predictions(self, id='None', special_identifier='', test=True, train=True):

        if test:
            filename = self.results_folder + "/predictions_test_split_" + id + special_identifier + '.csv'

            truth = (self.y_test > 0.5).float().sum(dim=1)

            dts = pd.DataFrame([self.mu.view(-1).detach().numpy(), truth.detach().numpy()]).T
            dts.columns = ["pred", "truth"]
            dts.to_csv(filename)

        # if train:
        #     filename = self.results_folder + "/predictions_train_split_" + id + '.csv'
        #     dts = pd.DataFrame([self.mu_train.view(-1).detach().numpy(), self.y_test.view(-1).detach().numpy(),
        #                         self.std_train.detach().view(-1).numpy()]).T
        #     # dts.columns = ["pred", "truth","std"]
        #     dts.to_csv(filename)