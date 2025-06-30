import torch, pyro, argparse, logging, yaml
import torch.nn.functional as F
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATv2Conv

class BayesianGNN(PyroModule):
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch, hid_ch, heads=4, concat=True)
        self.conv2 = GATv2Conv(hid_ch * 4, out_ch, heads=1, concat=False)
        self.conv1.lin_l.weight = PyroSample(dist.Normal(0.,1.).expand([hid_ch*4,in_ch]).to_event(2))
        self.conv1.lin_l.bias = PyroSample(dist.Normal(0.,1.).expand([hid_ch*4]).to_event(1))
        self.conv2.lin_l.weight = PyroSample(dist.Normal(0.,1.).expand([out_ch,hid_ch*4]).to_event(2))
        self.conv2.lin_l.bias = PyroSample(dist.Normal(0.,1.).expand([out_ch]).to_event(1))

    def forward(self, data, y=None):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        out = self.conv2(x, edge_index)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", pyro.distributions.Categorical(logits=out), obs=y)
        return out
