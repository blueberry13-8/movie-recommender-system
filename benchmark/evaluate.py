import argparse
import torch
import pandas as pd
from torch_geometric.nn import LGConv

from model.recsysgnn import RecSysGNN
from data.loader import data_loader
from model.metrics import compute_bpr_loss, get_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=3, type=int)    

    args = parser.parse_args()

    # Load model
    latent_dim = args.latent_dim
    num_layers = args.num_layers
    n_users = 943
    n_items = 1682
    model = RecSysGNN(latent_dim=latent_dim, num_layers=num_layers, num_users=n_users, num_items=n_items)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Load test data
    test_df = pd.read_csv(args.data_path)

    # Other necessary data loading or configurations if needed


    # Get model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Perform evaluation
    with torch.no_grad():
        _, out = model(train_edge_index)  # Assuming train_edge_index is available
        final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
        recall_at_10, precision_at_10 = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items, test_df, K=10)
        recall_at_20, precision_at_20 = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items, test_df, K=20)

    # Print the evaluation metrics
    print(f'Recall@10: {recall_at_10:.4f}')
    print(f'Precision@10: {precision_at_10:.4f}')
    print(f'Recall@20: {recall_at_20:.4f}')
    print(f'Precision@20: {precision_at_20:.4f}')
