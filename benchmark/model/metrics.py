import torch.nn.functional as F
import torch
import pandas as pd


def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K, device):
    # Get unique user IDs from the test set
    test_user_ids = torch.LongTensor(test_data['user_id_idx'].unique())

    # Compute the score of all user-item pairs
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))

    # Create a dense tensor of all user-item interactions
    i = torch.stack((
        torch.LongTensor(train_data['user_id_idx'].values),
        torch.LongTensor(train_data['item_id_idx'].values)
    ))
    v = torch.ones((len(train_data)), dtype=torch.float64)

    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=device).to_dense()

    # Mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))

    # Compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(),
                                             columns=['top_indx_' + str(x + 1) for x in range(K)])
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[
        ['top_indx_' + str(x + 1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID', 'top_rlvnt_itm']]

    # Measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id_idx',
                          right_on=['user_ID'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id_idx,
                                                                              metrics_df.top_rlvnt_itm)]

    # Compute recall and precision
    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id_idx']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)

    return metrics_df['recall'].mean(), metrics_df['precision'].mean()



def get_top_k_movies(user_id, user_Embed_wts, item_Embed_wts, K):
    """
    Get the top K movie indexes for a given user.

    Args:
        user_id (int): The user ID for which to get top K movies.
        user_Embed_wts (torch.Tensor): Embeddings for users.
        item_Embed_wts (torch.Tensor): Embeddings for items.
        K (int): Number of top movies to retrieve.

    Returns:
        list: Top K movie indexes for the given user.
    """
    # Compute the score of all user-item pairs
    user_scores = user_Embed_wts[user_id] @ item_Embed_wts.t()

    # Get the top K movie indexes
    topk_movie_indexes = torch.topk(user_scores, K).indices.cpu().numpy().tolist()

    return topk_movie_indexes

def get_metrics_for_user(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, user_id, K):
    """
    Compute recall and precision metrics for a given user.

    Args:
        user_Embed_wts (torch.Tensor): Embeddings for users.
        item_Embed_wts (torch.Tensor): Embeddings for items.
        n_users (int): Number of unique users.
        n_items (int): Number of unique items.
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.
        user_id (int): User ID for which to compute metrics.
        K (int): Number of top movies to consider for metrics.

    Returns:
        Tuple: Recall and Precision metrics.
    """
    # Get unique user IDs from the test set
    test_user_ids = torch.LongTensor(test_data['user_id_idx'].unique())

    # Compute the score of all user-item pairs
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))

    # Create a dense tensor of all user-item interactions
    i = torch.stack((
        torch.LongTensor(train_data['user_id_idx'].values),
        torch.LongTensor(train_data['item_id_idx'].values)
    ))
    v = torch.ones((len(train_data)), dtype=torch.float64)
    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=user_Embed_wts.device).to_dense()

    # Mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))

    # Get the top K movies for the specified user
    top_k_movies = get_top_k_movies(user_id, user_Embed_wts, item_Embed_wts, K)

    # Measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_user_data = test_data[test_data['user_id_idx'] == user_id]
    test_interacted_items = set(test_user_data['item_id_idx'].values)
    intersection_items = set(top_k_movies) & test_interacted_items

    # Compute recall and precision
    recall = len(intersection_items) / len(test_interacted_items) if len(test_interacted_items) > 0 else 0.0
    precision = len(intersection_items) / K

    return recall, precision

def get_movie_names_for_user(user_id, user_Embed_wts, item_Embed_wts, items, genres, K):
    """
    Get the names of the top K recommended movies for a given user.

    Args:
        user_id (int): User ID for which to get movie names.
        user_Embed_wts (torch.Tensor): Embeddings for users.
        item_Embed_wts (torch.Tensor): Embeddings for items.
        items (pd.DataFrame): DataFrame containing movie information.
        genres (pd.DataFrame): DataFrame containing genre information.
        K (int): Number of top movies to retrieve.

    Returns:
        list: Names of the top K recommended movies for the given user.
    """
    # Get the top K movie indexes for the user
    top_k_movies = get_top_k_movies(user_id, user_Embed_wts, item_Embed_wts, K)

    # Get movie names corresponding to the top K movie indexes
    top_k_movie_names = items.loc[top_k_movies, 'movieName'].tolist()

    return top_k_movie_names

