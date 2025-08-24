import torch
import torch.nn as nn


class AverageAlongDimension(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super(AverageAlongDimension, self).__init__()

    def forward(self, x):
        return x.mean(dim=self.dim)

class PrintSize(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class WeightedEndPoint(nn.Module):

    def __init__(self, seq1, seq2):
        super().__init__()
        self.seq1 = seq1
        self.seq2 = seq2

    def forward(self, x):
        return self.seq1(x) * self.seq2(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=99):
        super(PositionalEncoding, self).__init__()
        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # Add the positional encoding to the input tensor
        return x + self.pe.unsqueeze(0)

class MultiheadWrapper(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, x):
        x = x.transpose(0,1)
        output, _ = self.multihead_attn(x, x, x)
        return output.transpose(0,1)

class TransformerWrapper(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout = 0., forward_expansion = 10):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, forward_expansion * embedding_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_dim, embedding_dim),
        )
        self.dropout = dropout
    def forward(self, x):
        # first layer norm
        x = self.layernorm(x)
        # transpose for MHA
        y = x.transpose(0,1)
        # multihead attention
        output, _ = self.multihead_attn(y, y, y)
        # skip connection
        x = output.transpose(0,1) + x
        # droputout
        x = nn.Dropout(self.dropout)(x)
        # feed forward
        x = self.feed_forward(x)
        # 2nd layer norm
        x = self.layernorm2(x)
        return x
class MLPwithSkipWrapper(nn.Module):
    def __init__(self, layer_dim):
        super().__init__()
        self.layer = nn.Linear(layer_dim, layer_dim),
    def forward(self, x):
       return nn.ReLU(x + self.layer(x))

class NeuralNetwork():

    def __init__(self, type, emb_size, mid_layer, end_layer, output_size = 1, seq_len = 1,num_heads = 4):
        self.emb_size =emb_size
        self.mid_layer =mid_layer
        self.end_layer = end_layer
        self.output_size = output_size
        self.seq_len = seq_len
        self.num_heads = num_heads

        if type == "attention":
            self.model = torch.nn.Sequential(
                PositionalEncoding(self.emb_size, self.seq_len),
                TransformerWrapper(self.emb_size, self.num_heads),
                AverageAlongDimension(1),
                Reshape((-1, self.emb_size)),
                nn.ReLU(),
                nn.LayerNorm(self.emb_size),
                nn.Linear(self.emb_size, self.mid_layer),
                nn.ReLU(),
                nn.LayerNorm(self.mid_layer),
                nn.Linear(self.mid_layer, self.end_layer),
                nn.ReLU(),
                nn.LayerNorm(self.end_layer),
                nn.Linear(self.end_layer, self.output_size)
            )
        elif type == "fc":
            self.model = torch.nn.Sequential(
                nn.Flatten(start_dim =-2, end_dim=-1),
                nn.Linear(self.emb_size*seq_len, self.mid_layer),
                nn.BatchNorm1d(self.mid_layer),
                nn.ReLU(),
                nn.Linear(self.mid_layer, self.end_layer),
                nn.BatchNorm1d(self.end_layer),
                nn.ReLU(),
                nn.Linear(self.end_layer, self.output_size)
            )
        elif type == "fc+mean":
            self.model = torch.nn.Sequential(
                AverageAlongDimension(1),
                nn.Linear(self.emb_size, self.mid_layer),
                nn.BatchNorm1d(self.mid_layer),
                nn.ReLU(),
                nn.Linear(self.mid_layer, self.end_layer),
                nn.BatchNorm1d(self.end_layer),
                nn.ReLU(),
                nn.Linear(self.end_layer, self.output_size)
            )
        elif type == "linear":
            self.model = torch.nn.Sequential(
                Reshape((-1, self.emb_size*seq_len)),
                nn.Linear(self.emb_size*seq_len, self.output_size)

            )
        elif type == "linear+mean":
            self.model = torch.nn.Sequential(
                AverageAlongDimension(1),
                nn.Linear(self.emb_size, self.output_size))

        elif type == "linear+additive":
            self.model = torch.nn.Sequential(
                nn.Linear(self.emb_size, self.output_size),
                AverageAlongDimension(1)
                )

        elif type == "fc+additive":
            self.model = torch.nn.Sequential(
                nn.Linear(self.emb_size, self.mid_layer),
                nn.ReLU(),
                nn.Linear(self.mid_layer, self.end_layer),
                nn.ReLU(),
                nn.Linear(self.end_layer, self.output_size),
                AverageAlongDimension(1)
            )

        elif type == "fc+weighted":
            self.model1 = torch.nn.Sequential(
                nn.Linear(self.emb_size, self.mid_layer),
                nn.ReLU(),
                nn.Linear(self.mid_layer, self.end_layer),
                nn.ReLU(),
                nn.Linear(self.end_layer, self.output_size),
                AverageAlongDimension(1)
            )
            self.model2 = torch.nn.Sequential(
                Reshape((-1, self.emb_size * seq_len)),
                nn.Linear(self.emb_size*seq_len, 1),
                nn.Sigmoid()
            )
            self.model = WeightedEndPoint(self.model1, self.model2)
        else:
            raise NotImplementedError("Could not be elucidated.")

    def get_model(self):
        return self.model
