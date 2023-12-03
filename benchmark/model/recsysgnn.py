from torch_geometric.nn import LGConv
import torch.nn as nn
import torch

class RecSysGNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers,
        num_users,
        num_items,
    ):
        super(RecSysGNN, self).__init__()

        # Combine user and item embeddings into a single embedding matrix
        self.embedding = nn.Embedding(num_users + num_items, latent_dim)

        # Use LGConv for the graph convolutional layers
        self.convs = nn.ModuleList([LGConv(latent_dim) for _ in range(num_layers)])

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        emb0 = self.embedding.weight

        # Store intermediate embeddings in a list
        embs = [emb0]

        # Apply graph convolutional layers
        emb = emb0
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        # Aggregate intermediate embeddings by taking their mean
        out = torch.mean(torch.stack(embs, dim=0), dim=0)

        return emb0, out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        # Call the forward method to obtain embeddings
        emb0, out = self(edge_index)

        # Extract embeddings for users, positive items, and negative items
        user_emb, pos_item_emb, neg_item_emb = out[users], out[pos_items], out[neg_items]

        # Extract initial embeddings for users, positive items, and negative items
        user_emb0, pos_item_emb0, neg_item_emb0 = emb0[users], emb0[pos_items], emb0[neg_items]

        return user_emb, pos_item_emb, neg_item_emb, user_emb0, pos_item_emb0, neg_item_emb0
