# Import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix

import torch
import torch.optim as optim
import torch.utils.data as data

from src import mf_bpr, als, word2vec, metrics, datasets, utils

def main():
    # # Personalized Stock Recommender Systems
    # ## Prepare Data
    # ### Dummy Data

    # Read data
    def read_dummy():
        dummy_data = pd.read_csv("data/dummy.data", sep='\t', names = ['user_id', 'item_id',      
            'rating', 'timestamp'], engine = 'python')
        num_users = dummy_data.user_id.unique().shape[0]
        num_items = dummy_data.item_id.unique().shape[0]
        return dummy_data, num_users, num_items

    # ## Word2Vec
    # ### Dummy Data

    # Prep interactions
    def load_interactions_cbow(dummy_data : pd.DataFrame):
        interactions = {}
        for line in dummy_data.itertuples():
            user_index, item_index, time = line[1] - 1, line[2] - 1, line[4]
            interactions.setdefault(user_index, []).append((item_index, time))

        interactions = {k : sorted(v, key = (lambda pair : pair[1])) for k, v in interactions.items()}
        return {k : [x[0] for x in v] for k, v in interactions.items()}


    # Train test split
    def train_test_dummy_cbow(interactions : dict, window : int):
        train_targets, train_contexts = [], []
        test_targets, test_contexts = [], []

        # Iterate through every interaction
        for user_interactions in interactions.values():
            num_interactions = len(user_interactions)
            # Add to training data
            for i in range(window, num_interactions - 1):
                train_targets.append(user_interactions[i])
                train_contexts.append([user_interactions[j] for j in np.arange(i - window, i)])
            # Add to testing data
            test_targets.append(user_interactions[num_interactions - 1])
            test_contexts.append([user_interactions[j] for j 
                in np.arange(num_interactions - 1 - window, num_interactions - 1)])
            
        return train_targets, train_contexts, test_targets, test_contexts


    # Define evaluator
    def evaluate_ranking_cbow(net, ngrams_dataloader_test, num_items, num_users):
        ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
        item_ids = list(range(num_items))
        
        for _, (targets, contexts) in enumerate(ngrams_dataloader_test):
            scores = net(contexts).tolist()
            for u, row in enumerate(scores):
                item_scores = list(zip(item_ids, row))
                ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
                ranked_items[u] = [r[0] for r in ranked_list[u]]
            
                temp = metrics.hit_and_auc(ranked_items[u], test_targets[u], 50)
                hit_rate.append(temp[0])
                auc.append(temp[1])
        return np.mean(np.array(hit_rate)), np.mean(np.array(auc))

    # Prepare data
    window = 10

    dummy_data, num_users, num_items = read_dummy()
    sorted_interactions = load_interactions_cbow(dummy_data)
    train_targets, train_contexts, test_targets, test_contexts = train_test_dummy_cbow(sorted_interactions, window)


    # Prepare dataset and model
    ngrams_train = data.TensorDataset(torch.from_numpy(np.array(train_targets)), 
        torch.from_numpy(np.array(train_contexts)))
    ngrams_dataloader = data.DataLoader(dataset = ngrams_train, batch_size = 1024, 
        shuffle = True, num_workers = 4)
    ngrams_test = data.TensorDataset(torch.from_numpy(np.array(test_targets)), 
        torch.from_numpy(np.array(test_contexts)))
    ngrams_dataloader_test = data.DataLoader(dataset = ngrams_test, batch_size = 1024, 
        shuffle = False, num_workers = 4)

    embedding_dim, num_epochs = 30, 20
    loss = torch.nn.NLLLoss()
    cbow_net = word2vec.CBOW(num_items, embedding_dim, window)
    optimizer = optim.Adam(cbow_net.parameters(), lr = 0.01)


    # Train and evaluate the model
    hit_rate_list = []
    auc_list = []
    for epoch in range(num_epochs):
        accumulator, l = utils.Accumulator(2), 0.

        # Train each batch
        cbow_net.train()
        for _, (targets, contexts) in enumerate(ngrams_dataloader):
            optimizer.zero_grad()

            log_probabilities = cbow_net(contexts)

            total_loss = loss(log_probabilities, targets)
            total_loss.backward()
            optimizer.step()
            accumulator.add(total_loss, targets.shape[0])

        # Evaluate
        cbow_net.eval()
        hit_rate, auc = evaluate_ranking_cbow(cbow_net, ngrams_dataloader_test, num_items, num_users)
        hit_rate_list.append(hit_rate)
        auc_list.append(auc)

        print(f"Epoch {epoch}:\n\tloss = {accumulator[0]/accumulator[1]}\n\thit_rate = {hit_rate}\n\tauc = {auc}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()