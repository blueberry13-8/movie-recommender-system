import torch
import random
import pandas as pd


def data_loader(data, batch_size, n_usr, n_itm, device):
    """
    Custom data loader for generating positive and negative samples for training.

    Args:
        data (pd.DataFrame): Training data containing user-item interactions.
        batch_size (int): Number of samples to generate in each batch.
        n_usr (int): Number of unique users.
        n_itm (int): Number of unique items.

    Returns:
        Tuple of three PyTorch LongTensor: (user_ids, positive_item_ids, negative_item_ids)
    """

    def sample_neg(x):
        """
        Sample a negative item not interacted by the user.
        """
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    # Group by user and get a list of interacted items for each user
    interacted_items_df = data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()

    # Sample users for the batch
    if n_usr < batch_size:
        users = [random.choice(range(n_usr)) for _ in range(batch_size)]
    else:
        users = random.sample(range(n_usr), batch_size)
    users.sort()

    users_df = pd.DataFrame(users, columns=['user_id_idx'])

    # Merge with interacted items DataFrame to get positive items
    interacted_items_df = pd.merge(interacted_items_df, users_df, how='right', left_on='user_id_idx', right_on='user_id_idx')

    pos_items = interacted_items_df['item_id_idx'].apply(lambda x: random.choice(x)).values
    # Generate negative items for each user in the batch
    neg_items = interacted_items_df['item_id_idx'].apply(lambda x: sample_neg(x)).values

    return (
        torch.LongTensor(users).to(device),
        torch.LongTensor(pos_items).to(device) + n_usr,
        torch.LongTensor(neg_items).to(device) + n_usr
    )