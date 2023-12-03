import argparse
import torch
import pandas as pd
from torch_geometric.nn import LGConv

from model.recsysgnn import RecSysGNN
from model.metrics import get_metrics, get_metrics_for_user, get_movie_names_for_user


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('edge_index_path', type=str)
    parser.add_argument('prediction_type', type=str)
    # parser.add_argument('data_path', type=str)
    parser.add_argument('--user_id', type=int, default=5)
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)

    args = parser.parse_args()

    # Load model
    latent_dim = args.latent_dim
    num_layers = args.num_layers
    n_users = 943
    n_items = 1539
    model = RecSysGNN(latent_dim=latent_dim, num_layers=num_layers, num_users=n_users, num_items=n_items)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Get model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load data
    train_df = pd.read_csv('./benchmark/data/train_df.csv')
    test_df = pd.read_csv('./benchmark/data/test_df.csv')
    train_edge_index = torch.load(args.edge_index_path).to(device)

    if args.prediction_type == 'single':
        user_id = args.user_id
        with torch.no_grad():
            test_df = pd.read_csv('./benchmark/data/test_df.csv')
            genres = pd.read_csv('./data/interim/u.genre', sep='|', header=None)
            ind2genre = genres[0].tolist()
            genre2ind = {g: i for i, g in enumerate(ind2genre)}

            items = pd.read_csv('./data/interim/u.item', sep='|', encoding='latin-1',
                            names=['movieId', 'movieName', 'releaseDate', 'videoDate', 'url'] + ind2genre)

            _, out = model(train_edge_index)
            final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
            recall, presicion = get_metrics_for_user(final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, user_id, K=20)

            print(f'Recall@20: {presicion:.4f}')
            print(f'Precision@20: {recall:.4f}')
            print()
    
            recommended_movies = get_movie_names_for_user(user_id, final_user_Embed, final_item_Embed, items, ind2genre, 20)

            # Print the recommended movies
            print(f"Top {20} movies recommended for User {user_id}:")
            for i, movie_name in enumerate(recommended_movies, 1):
                print(f"{i}. {movie_name}")
        
    elif args.prediction_type == 'multiple':
        # Perform evaluation
        with torch.no_grad():
            _, out = model(train_edge_index)
            final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
            recall_at_10, precision_at_10 = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, 10, device)
            recall_at_20, precision_at_20 = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, 20, device)

        # Print the evaluation metrics
        print(f'Recall@10: {recall_at_10:.4f}')
        print(f'Precision@10: {precision_at_10:.4f}')
        print(f'Recall@20: {recall_at_20:.4f}')
        print(f'Precision@20: {precision_at_20:.4f}')
