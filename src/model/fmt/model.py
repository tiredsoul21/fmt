""" FeatureManifoldTransformer model implementation """
import math

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
from src.model.fmt.model_parts import Block, LayerNorm

class FMTConfig():
    """ Config for the Differential Item Function Neural Network."""
    def __init__(self):
        # Cat Keys to size of the categories
        self.demos = {} # key: category, value: size
        self.knows = {} # key: category, value: size
        self.bias = False
        self.lambda_reg = 0.001
        self.noise_std = 0.1
        self.dropout = 0.0
        self.num_layers = 2
        self.num_embd = 2
        self.num_heads = 2
        self.block_size = None
        self.vocab_size = None
        self.target = ''

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

class DatasetDict(Dataset):
    """ Dataset class to handle the data for the neural network. """
    def __init__(self, pd_data: pd.DataFrame, config: FMTConfig):
        self.pd_data = pd_data
        self.config = config

    def __len__(self):
        return len(self.pd_data)

    def __getitem__(self, idx):
        # Get all categories in demo and know as tensor
        return_tensors = {}
        for key in self.config.demos.keys():
            return_tensors[key] = torch.tensor(self.pd_data[key].iloc[idx], dtype=torch.long)
        for key in self.config.knows.keys():
            return_tensors[key] = torch.tensor(self.pd_data[key].iloc[idx], dtype=torch.long)
        key = self.config.target
        return_tensors[key] = torch.tensor(self.pd_data[key].iloc[idx], dtype=torch.long)
        return return_tensors

    @staticmethod
    def generate_labels(p_inputs: dict, p_groups: list):
        """
        Generate labels based on the specified isolation group.
        :param p_inputs: Dictionary of input tensors
        :param p_groups: List of group names
        :return: List of label strings, with comma-seperated values of data
        """
        labels = []
        for i in range(len(next(iter(p_inputs.values())))):
            label = ', '.join(str(p_inputs[group][i].item()) for group in p_groups)
            labels.append(label)
        return labels

class FeatureManifoldTransformer(nn.Module):
    """ FeatureManifoldTransformer model """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.vocab_size = sum(config.demos.values())
        self.config.block_size = sum(config.knows.values())
        self.idx_labels = self.make_index_map()

        self.demo_embed_tbls = nn.ModuleDict({key: nn.Embedding(config.demos[key], config.num_embd)
                                              for key in config.demos.keys()})

        self.transformer = nn.ModuleDict({
            "know_embed_tbl": nn.Embedding(self.config.block_size, config.num_embd),
            "blocks": nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            "ln_f": LayerNorm(config.num_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.num_embd, self.config.vocab_size, bias=False)

        # Collapses demo into a single embedding (count of demos keys x vocab_size)
        demo_table_count = len(config.demos)
        # self.combiner = nn.Linear(demo_table_count* self.config.vocab_size, 1, bias=False)
        self.combiner = nn.Linear((demo_table_count + 1) * self.config.vocab_size, 1, bias=False)

        # with weight tying https://paperswithcode.com/method/weight-tying
        # Perform weight tying
        self.tie_weights()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

        # report number of parameters
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def tie_weights(self):
        """ Tie the weights of the embedding tables to lm_head """
        start_idx = 0
        print(self.lm_head.weight)
        for key, embed_tbl in self.demo_embed_tbls.items():
            end_idx = start_idx + self.config.demos[key]
            embed_tbl.weight = nn.Parameter(self.lm_head.weight[start_idx:end_idx])
            start_idx = end_idx

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.know_embed_tbl.weight.numel()
        return n_params

    def make_index_map(self) -> dict:
        """Create a dictionary mapping the indices to the labels."""
        label_idx = 0
        idx_labels = {}
        for key in self.config.demos.keys():
            idx_labels[key] = label_idx
            label_idx += 1
        for key in self.config.knows.keys():
            idx_labels[key] = label_idx
            label_idx += 1
        idx_labels[self.config.target] = label_idx
        return idx_labels

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, p_data: dict, std_override= None):
        """ Forward pass of the FeatureManifoldTransformer model """
        # Get the embeddings for the tokens in the context (batch, n_embd)
        demo_embs = [self.demo_embed_tbls[key](p_data[key]) for key in self.config.demos.keys()]
        # Stack the embeddings
        demo_emb = torch.stack(demo_embs, dim=1)

        # Get the knowledge embedding
        know_key = list(self.config.knows.keys())[0]
        know_emb = self.transformer.know_embed_tbl(p_data[know_key])

        # ############################################

        # # Add noise to the embeddings
        # noise = torch.normal(0, self.config.noise_std, size=demo_emb.size()).to(demo_emb.device)
        # x = demo_emb + know_emb.unsqueeze(1) + noise
        # print(x.shape)

        # know_key = list(self.config.knows.keys())[0]
        # know_emb = self.transformer["know_embed_tbl"](p_data[know_key])
        # know_emb = know_emb.unsqueeze(1)  # Ensure the correct shape

        # all_emb = torch.cat((demo_emb, know_emb), dim=1)

        # Add noise to the embeddings
        # noise = torch.normal(0, self.config.noise_std, size=all_emb.size()).to(all_emb.device)
        # x = all_emb + noise

        # ###########################################

        # Concatenate the embeddings
        all_emb = torch.cat((demo_emb, know_emb.unsqueeze(1)), dim=1)
        # all_emb = all_emb.view(all_emb.size(0), -1)  # Flatten the embeddings

        # Add noise to the embeddings
        noise = torch.normal(0, self.config.noise_std, size=all_emb.size()).to(all_emb.device)
        if std_override is not None and std_override != 0:
            noise = torch.normal(0, std_override, size=all_emb.size()).to(all_emb.device)
        elif std_override == 0:
            noise = noise * 0
        x = all_emb + noise

        # Pass through the transformer blocks
        for block in self.transformer["blocks"]:
            x = block(x)

        # Save intermediate embeddings
        # x = x + noise
        intermediate_embeddings = x

        # Layer normalization
        x = self.transformer["ln_f"](x)
        return_logits = self.lm_head(x)

        # Flatten the embeddings on last two dimensions
        return_logits = return_logits.view(return_logits.size(0), -1)

        return_logits = self.combiner(return_logits).squeeze(-1)
        return_logits = torch.sigmoid(return_logits)

        return_loss = None

        if p_data[self.config.target] is not None:
            # If we are given some desired targets also calculate the loss
            return_loss = nn.functional.binary_cross_entropy(return_logits, p_data[self.config.target].float())
            # Add regularization loss
            return_loss += self.embed_regularization(self.config.lambda_reg)

        return return_logits, return_loss, intermediate_embeddings

    def embed_regularization(self, lambda_reg: float = 0.001):
        """
        Regularization function for the embeddings
        Acts as a L2 regularization (pulls the embeddings closer to 0)
        :param ref_model: DIFNeuralNetwork model
        :param lambda_reg: float value for the regularization
        """
        reg_loss = 0
        for embedding in self.demo_embed_tbls.values():
            reg_loss += torch.sum(embedding.weight ** 2)
        reg_loss += torch.sum(self.transformer["know_embed_tbl"].weight ** 2)

        return lambda_reg * reg_loss
