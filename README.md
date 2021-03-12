# **Personalized Stock Recommender Systems**

## **Contents**
RecSys for Banking and Financial Services:
1. Introduction
2. Goal
3. Drawbacks of Current Methods
4. Datasets
5. Method 1: Matrix Factorization with Bayesian Personalized Ranking
6. Method 2: Alternating Least Squares
7. Method 3: Word2Vec/CBOW

Training and Evaluating RecSys Models:
1. Dummy Dataset
2. Representative Dataset
<br/><br/>

# RecSys for Banking and Financial Services

## **Introduction**
Financial institutions are seriously looking to machine learning to provide tailored services and customized experiences to their customers.  Recommender systems (RecSys) are one class of algorithms to solve this problem.  These models are typically used in the realm of entertainment and e-commerce to recommend media or things to purchase, respectively. The paper [Recommender Systems for Banking and Financial Services](http://ceur-ws.org/Vol-1905/recsys2017_poster13.pdf) by Andrea Gigli, Fabrizio Lillo, and Daniele Regoli extends recommender systems to FinTech.

My project is to replicate this paper to the best of my ability.  Despite not having the data the author's do, the replicated models which I produce perform quite well on the data I do have.
<br/><br/>

## **Goal**
The question we are trying to answer is: given a portfolio of an investor's stocks, what stock is the investor most likely to invest in next?  This question has major applications to trading platforms like Fidelity and Robinhood, which could personalize recommendations to investors.
<p align="center">
<br/><br/>
<img src="images/goal.png" alt = "goal" width="75%"/>
<br/><br/>
</p>
In the graphic above, we have an investor on the left who has invested in tech companies like IBM, Intel, AMD and Google, but not in automotive companies like ford.  This information is given to a model which then outputs a list of stocks which it believes the investor is most likely to purchase.  We see that NVIDIA and Apple are at the top while General Motors is quite low.
<br/><br/>

## **Drawbacks of Current Methods**
Why is a new recommendation system needed, though, in the first place? Well, recommender systems in FinTech are relatively new, just becoming prevalent in the past five years or so:
- Financial institutions still typically conduct their own research  and provide opinions to investors
- At publication, many methods in the literature base their recommendations on broker research and news using NLP
- These models take a long time to train and are costly

Moreover, the literature tends towards explicit, un-personalized recommenders.  Explicit means that the information collected directly reflects explicit opinions of the investor.  Un-personalized means that the recommender provides the same recommendations to everyone, such as a popularity-based system.  Both of these things are unideal because explicit information is not always necessary and an un-personalized system is more disconnected from investors.  What we want is an implicit, personalized  recommender that is only given "purchased"/"not purchased" information.  This will lead to a happy investor and the firm implementing the recommender system to make more money, as illustrated in the graphic here. 
<p align="center">
    <img src="images/personalized.png" alt = "personalized" width="50%"/>
    <br/><br/>
</p>

## **Recommender Systems to the Rescue**
At the 2017 ACM Recommender Systems conference, Gigli, Lillo, and Regoli showed that an *implicit* recommender system can predict preferences of users (investors) and the items (stocks) they purchase.  The showcased three different RecSys methods:
- Matrix factorization with Bayesian Personalized Ranking (BPR)
- Alternating Least Squares (ALS)
- Word2Vec/Continuous Bag of Words

The paper compares these algorithms against popularity methods that base their predictions solely on the popularity of different items (completely unpersonalized). 
<p align="center">
<br/><br/>
<img src="images/recsys_poster.png" alt = "recsys_poster" width="66%"/>
<br/><br/>
</p>

## **Datasets**
The data used for these recommendation systems is an interaction matrix between investors and the stocks they purchase.  More specificaly, we need a relation where each record is a transaction that has
- the investor identification number (```int```)
- the stock identification number (```int```)
- the timestamp of that transaction (```int```)

The interactions matrix itself should end up boiling down to somelike like the table below.
<p align="center">
<br/><br/>
<img src="images/desired_data.png" alt = "desired_data" width="66%"/>
<br/><br/>
</p>


### **Ideal Data**
The authors of the paper obtain this data from a European bank, where about 200,000 clients make 1.3 million transaction total.  Unfortunately, this data is proprietary and not available to us.  In this data's stead, we use two other datasets: a dummy dataset for testing the model and a representative dataset of transactions collected from UC Irvine.

### **Dummy Data**
To test to see if our model works, we will use the [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) dataset.  This dataset contains 100,000 records of about 1,000 different users each interacting with, on average, 100 movies from a population of 1,600 movies.  So, our interaction matrix will have a shape of approximately 1,000 rows by 1,600 columns.

### **Representative Dataset**
Orginially, I was going to use 13F forms submitted by hedge funds in Q4 2020 instead of users/stocks; however, the thousands of hedge funds I looked at invested from too large of a popultion of stocks.  This resulted in the interaction matrix becoming too scarce to make meaningful predictions.  Time permitting, I would like to continue to work towards using 13F forms, as my models were effective when using a small subset of hedge funds with similar investment strategies.

Instead, per the TA's permission, I will use a synthetic, representative dataset of individual transactions from [UC Irvine](https://archive.ics.uci.edu/ml/datasets/online+retail).  This dataset contains approximately 540,000 records.
<br/><br/>

## **Method 1: Matrix Factorization with BPR [(Rendle et al., 2012)](https://arxiv.org/pdf/1205.2618.pdf)**
Matrix factorization (MF) is used because it captures the low-rank structure of linear investor-stock interactions.  In the figure below, we let $m, n, k \in \mathbb{N}$, where $m$ is the number of investors, $n$ is the number of stocks, and $k$ is the number of latent factors in $P$ and $Q$.
<p align="center">
<img src="images/matrix_factorization.png" alt = "matrix_factorization" width="66%"/>
<br/><br/>
</p>

The general model of MF is that there is an investor/stock interaction matrix $R$ which can be broken down into two latent matrices $P$ and $Q$.  MF finds these latent matrices using mean squared error and an optimizer such as Adam and uses it to predict unknown ratings.  However, in our case, since we are using *implicit* information, it is imperative that we use an optimization criterion such as Bayesian Personalized Ranking (BPR) over pairs of stocks for a particular investor when updating our model's parameters.  In this implicit scenario, the mulitplication of $P$ and $Q$ won't result in an explicit reconstruction of $R$, but rather a list of scores for each stock which we can then use to rank preferences.

To illustrate BPR, first let $I$ denote all stocks and $I^+$ denote purchased stocks.  Then, BPR is defined for pairs of stocks (per investor $u$) in the set
$$D:=\{(u, i, j)\ |\ i \in I_{u}^{+} \wedge j \in I \setminus I_{u}^{+} \}$$
Let's further define $\hat{y}$ as the binary prediction of "purchased" (1) or "not purchased" (0), $\lambda$ as the regularization hyperparameter, and $\Theta$ as the learned parameters.  Then BPR loss with L2-regularization is defined to be
$$\text{BPRLoss} := \sum_{u, i, j \in D}\ln(\sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta ||\Theta||^2$$
where $\sigma$ is the sigmoid function.
<br/><br/>

### **Model Implementation in PyTorch**
The above logic is written in PyTorch in the file [mf_bpr.py](src/mf_bpr.py).  We first start with the model itself, which has two embeddings matrices as defined using ```nn.Embedding``` and then initialized via a normal distribution.  During the forward propagation step, ids corresponding to the investor and ids of the stocks they purchased are supplied.  The embeddings of these ids are obtained and the dot product of them are multiplied together to compute the scores.
```python
class MF_BPR(nn.Module):
    def __init__(self, investor_num: int, stock_num: int, latent_factors: int):
        """
        Initializes a matrix factorization model that is meant to be used in
        conjunction with Bayesian Personalized Recommendation loss.

        Parameters
        ----------
        investor_num (int) - number of investors\n
        stock_num (int) - number of stocks\n
        latent_factors (int) - number of latent factors
        """
        super(MF_BPR, self).__init__()
        self.embed_investor = nn.Embedding(investor_num, latent_factors)
        self.embed_stock = nn.Embedding(stock_num, latent_factors)

        nn.init.normal_(self.embed_investor.weight, std=0.01)
        nn.init.normal_(self.embed_stock.weight, std=0.01)

    def forward(self, investors: torch.Tensor, stocks: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        investors (torch.Tensor) - investor ids\n
        stocks (torch.Tensor) - ids of stocks that the investors purchased\n

        Output
        ------
        scores  (torch.Tensor) - scores of stocks that the investors may purchase next
        """
        investor = self.embed_investor(investors)
        stock_positive = self.embed_stock(stocks)
        scores = (investor * stock_positive).sum(dim=-1)

        return scores
```
The Bayesian Personalized ranking loss which accompanies the loss is constructed outside of the class.  Given score tensors of (investors, num_stocks), we aim to maximize the distance between the positive and negative scores:
```python
def BPR_Loss(positive : torch.Tensor, negative : torch.Tensor) -> torch.Tensor:
    """
    Given postive and negative examples, compute Bayesian Personalized ranking loss
    """
    distances = positive - negative
    loss = - torch.sum(torch.log(torch.sigmoid(distances)), 0, keepdim=True)

    return loss
```
<br/><br/>

## **Method 2: Alternating Least Squares [(Zhou et al., 2008)](https://doi.org/10.1007/978-3-540-68880-8_32)**
Alternating least squares is a distributed analog to matrix factorization.  It improves upon original matrix factorization by taking in implicit information and iteratively alternating between optimizing the latent investor matrix and fixing the latent stock matrix and vice versa.  This alternation in illustrated in the graphic on the right, where only one matrix is learned at a time.
<p align="center">
<img src="images/als.png" alt = "als" width="66%"/>
<br/><br/>
</p>

The benifit of fixing one matrix at a time is that it enables an analytical solution to the optimization problem.  Consider a mean squared error with L2-regularization:
$$\text{MSE} = \sum_{u, i}\left(R_{ui} - \hat{R}_{ui}\right)^2 + \lambda_\Theta||\Theta||^2$$
If $P$ is fixed, then the analaytical solution to the optimization problem is
$$Q = R^TP\left(P^TP + \lambda I_k\right)^{-1}$$
If $Q$ is fixed, then the analytical solution to the optimization problem is
$$P = R^TQ\left(Q^TQ + \lambda I_k\right)^{-1}$$
where $I_k$ is the identity matrix of dimension $k$.

Computing these analytical solutions is extremely fast compared to learning the latent factors in a approximate, gradient descent-like fashion.  Indeed, ALS is meant to be used in a distributed environment with Apache Spark in large scale environments for fast performance.
<br/><br/>

### **Model Implementation in NumPy**
The above logic is written in numpy in the file [als.py](src/als.py).  We first start with the model itself, which, like MF, has two embeddings matrices and then initialized via a normal distribution.  During the training step, we first assume the stock embedding matrix to be fixed and compute the investor embedding matrix via the first analytical solution above; then, we assume the investor embedding matrix to be fixed and compute the stock embedding via the second analytical solution above.  During predictions, we simply need to multiple to two embedding matricies together and obtain the corresponding (investor, stock) predictions.
```python
class ALS():
    def __init__(self, investor_num: int, stock_num: int, latent_factors: int,
                 train_data: np.ndarray, reg: float):
        """
        An Alternating Least Squares model which alternates between freezing the
        investor embedding matrix and the stock embedding matrix.

        Parameters
        ----------
        investor_num (int) - number of investors\n
        stock_num (int) - number of stocks\n
        latent_factors (int) - number of latent factors\n
        train_data (np.ndarray) - training data of shape (investor_num, stock_num)\n
        reg (float) - regularization factor
        """
        self.n_factors = latent_factors
        self.embed_investor = np.random.random((investor_num, latent_factors))
        self.embed_stock = np.random.random((stock_num, latent_factors))
        self.train_data = train_data
        self.reg = reg

    def train(self) -> None:
        """
        Train the ALS model by first fixing the stock embedding matrix and learning
        the investor embeddings and then the other way around.
        """
        # Train investor embeddings
        A_1 = np.matmul(self.embed_stock.T, self.embed_stock) + \
            np.eye(self.n_factors) * \
            self.reg  # (latent_factors, latent_factors)

        b_1 = np.matmul(self.train_data, self.embed_stock)
        self.embed_investor = np.matmul(b_1, np.linalg.inv(A_1))

        # Train stock embeddings
        A_2 = np.matmul(self.embed_investor.T, self.embed_investor) + \
            np.eye(self.n_factors) * \
            self.reg  # (latent_factors, latent_factors)
        b_2 = np.matmul(self.train_data.T, self.embed_investor)
        self.embed_stock = np.matmul(b_2, np.linalg.inv(A_2))

    def predict(self, user_ids: list, item_idxs: list) -> list:
        """
        Make matrix predictions using the ALS model
        """
        predictions = np.matmul(self.embed_investor, self.embed_stock.T)
        return [predictions[user_ids[i], item_idxs[i]] for i in range(len(user_ids))]
```
<br/><br/>

## **Method 3: Word2Vec/CBOW [(Mikolov et al. 2013a)](https://arxiv.org/abs/1301.3781)**
The Word2Vec model is typically used in NLP to create word embeddings for different words in a corpus of text over different documents.  Here, investors' portfolios are documents, and each stock is a word $w$ from a vocabulary $\mathcal{V}$.  There are two variants of Word2Vec, Skip-Gram and Continuous Bag of Words (CBOW), but the variant we're interested in is CBOW, as it allows us to predict whether one purchased stock is in an investor's portfolio.  The graphic on the left illustrates the objective of this model: given a portfolio of stocks like IBM, Intel, AMD, and Texas Instruments, we want to maximize the probability of another purchased tech stock like NVIDIA and minimize the probability of some random stock like General Motors.
<p align="center">
<img src="images/cbow_obj.png" alt = "cbow_obj" width="50%"/>
<br/><br/>
</p>

Let $\mathcal{W}_o$ be a set of context words (in a window of size $m$) and $u_c$, $v_o \in \mathbb{R}^d$ represent $d$-dimensional target word and context word embeddings, respectively.  Then, the conditional probability of a target word is simply the softmax of that target word:
$$\Pr(w_c\ |\ \mathcal{W}_0) = \frac{e^{u_c^Tv_o}}{\sum_{i \in \mathcal{V}}e^{u_i^Tv_o}}$$
The loss function for CBOW can be derived from the maximim likelihood estimation, where $L$ is the length of the portfolio and a word at time $t$ is $w^{(t)}$:
$$\text{CBOW-OPT} := \prod_{t = 1}^L\Pr\left(w^{(t)}\ |\ w^{(t-m)},\ldots, w^{(t-1)}, w^{(t+1)},\ldots,w^{(t+m)}\right)$$
<br/><br/>

### **Model Implementation in PyTorch**
The above logic is written in PyTorch in the file [word2vec.py](src/word2vec.py).  We first start with the model itself, which has...
- an embedding matrix of size (```vocab_size```, ```embedding_dim```) defined using ```nn.Embedding```
- an input linear layer defined using ```nn.Linear```.  It takes in ```context_size * embedding_dim``` number of inputs and produces ```128``` outputs for the hidden layer.
- a hidden layer defined using ```nn.Linear```.  It takes in ```128``` hidden inputs and produces ```vocab_size``` number of outputs.
  
During the forward propagation step, embeddings of the context stocks are computed and then passed into the multi-layer perceptron (MLP).  The output of this MLP is then given to ```F.log_softmax```, which computes the probabilities of each of the stocks.
```python
class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int):
        """
        Initialize a CBOW model. Adapted from https://srijithr.gitlab.io/post/word2vec/.

        Parameters
        ----------
        vocab_size (int) - the size of all stocks under consideration\n
        embedding_dim (int) - the size of the embedding dimension\n
        context_size (int) - the number of words used as context for prediction
        """
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.context_size = context_size
        self.embedding_dim = embedding_dim

    def forward(self, context_words: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        context_words (torch.Tensor) - a tensor of words ids used as context for prediction of
            the target

        Output
        ------
        log_probabilities (torch.Tensor) - a tensor of the probabilities of each word
            in the vocabulary
        """
        embeddings = self.embeddings(context_words).view((context_words.shape[0],
            self.context_size * self.embedding_dim))
        out1 = F.relu(self.linear1(embeddings))
        out2 = self.linear2(out1)
        log_probabilities = F.log_softmax(out2, dim=1)
        return log_probabilities
```
As we'll see later, the loss funciton being used in conjunction with ```CBOW``` is ```nn.NLLLoss```, which is the negative log likelihood loss used to train a classification problem with multiple classes.
<br/><br/>

# Training and Evaluating RecSys Models
First, let's import the relevant packages that will be used throughout:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix

import torch
import torch.optim as optim
import torch.utils.data as data

from src import mf_bpr, als, word2vec, metrics, datasets, utils
```
To evaluate our models, we need to compute the hitting rate and AUC.  The hitting rate at a given position $l$ for each user represents whether the recommended item is included in the top $l$ ranked list.  It is defined as 
$$\text{Hit(l)} = \frac{1}{m}\sum_{u \in \mathcal{U}} \mathbf{1}(rank_{u, g_u} \leq l),$$
where $\mathbf{1}$ denotes the indicator function, $rank_{g,g_u}$ denotes the ranking of the ground truth item $g_u$ of the user $u$ in the recommendation list, $m$ is the number of users, and $\mathcal{U}$ is the set of all users.  AUC, on the other hand is a measurement of the area of the receiving operating characteristic curve; it represents the probability that the score assigned to a true positive sample is higher than the score assigned to a false positive sample.  The computation of AUC piggybacks on hitting rate with the following definition
$$\text{AUC} = \frac{1}{m}\sum_{u \in \mathcal{U}}\frac{1}{|\mathcal{I}\setminus S_u|}\sum_{j \in I \setminus S_u}\mathbf{1}(rank_{u, g_u} \leq l),$$
where $\mathcal{I}$ is the item set and $S_u$ are the items a user $u$ has already interacted with.  These two metrics are defined in [metrics.py](src/metrics.py):
```python
def hit_and_auc(rankedlist, test_item, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if int(val) == int(test_item)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if int(val) == int(test_item)]

    max_num = len(rankedlist) - 1
    auc = 1.0 * (max_num - hits_all[0][0]) / \
        max_num if len(hits_all) > 0 else 0
    return len(hits_k), auc
```
<br/><br/>

## **Model Efficacy Check with Dummy Dataset**
To start, let's read in the [dataset](data/dummy.data):
```python
# Read data
def read_dummy():
    dummy_data = pd.read_csv("data/dummy.data", sep='\t', 
        names = ["user_id", "item_id", "rating", "timestamp"], engine = "python")
    num_users = dummy_data.user_id.unique().shape[0]
    num_items = dummy_data.item_id.unique().shape[0]
    return dummy_data, num_users, num_items
```
<br/><br/>

### **Matrix Factorization with BPR**
#### **Loading Data**
We first start off with a custom ```Dataset``` object that is created with the PyTorch semantics and is defined in [datasets.py](src/datasets.py).  This dataset loads corresponding user indices, item indices, and interactions of each user.  When we index into the datset, we proced the corresponding user, item, and some negative item which the user has not interacted with yet.
```python
class DummyDataset(data.Dataset):
    def __init__(self, user_idxs : np.ndarray, item_idxs : np.ndarray, interactions : dict,
        num_items : int):
        assert user_idxs.shape[0] == item_idxs.shape[0]

        self.n_samples = user_idxs.shape[0]

        self.user_idxs = torch.from_numpy(user_idxs)
        self.item_idxs = torch.from_numpy(item_idxs)

        self.interactions = interactions
        self.all = set([i for i in range(num_items)])
   
    def __getitem__(self, index):
        neg_items = list(self.all - set(self.interactions[int(self.user_idxs[index])]))
        neg_idx = random.randint(0, len(neg_items) - 1)
        return self.user_idxs[index], self.item_idxs[index], neg_items[neg_idx]

    def __len__(self):
        return self.n_samples
```
Now, let's define a model to split our data into train and test sets, where the test set is comprised of the most recent rating of each user.
```python
def train_test_dummy_bpr(dummy_data : pd.DataFrame, num_users : int, num_items : int):
    train_items, test_items, train_list = {}, {}, []

    # Iterate through every line in the raw data
    for line in dummy_data.itertuples():
        u, i, rating, time = line[1], line[2], line[3], line[4]
        train_items.setdefault(u, []).append((u, i, rating, time))
        if u not in test_items or test_items[u][2] < time:
            test_items[u] = (i, rating, time)
        
    # Iterate through every user and add their samples, sorted by timestamp, to the train 
    # list
    for u in range(1, num_users + 1):
        train_list.extend(sorted(train_items[u], key = (lambda x : x[3])))

    test_data = [(key, *value) for key, value in test_items.items()]

    train_data = [item for item in train_list if item not in test_data]
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    return train_data, test_data
```
Let's define another function which loads the user and item indices (zero based) and all the interactions between users and items.
```python
def load_dummy_bpr(dummy, num_users, num_items):
    users, items, scores = [], [], []
    interactions = {}
    for line in dummy.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = 1 # implicit

        users.append(user_index)
        items.append(item_index)
        scores.append(score)

        interactions.setdefault(user_index, []).append(item_index)

    return users, items, scores, interactions
```
Given these functions, we can finally ready the training and testing data.
```python
# Ready dummy data
dummy_data, num_users, num_items = read_dummy()
train_dummy, test_dummy = train_test_dummy_bpr(dummy_data, num_users, num_items)

# Training data
train_users, train_items, train_ratings, interactions = load_dummy_bpr(train_dummy,    
    num_users, num_items)
train_dummy_dataset = datasets.DummyDataset(np.array(train_users), np.array(train_items),
    interactions, num_items)
train_dataloader = data.DataLoader(dataset = train_dummy_dataset, batch_size = 1024, 
    shuffle = True, num_workers = 4)

# Test data
_, _, _, test_interactions = load_dummy_bpr(test_dummy, 
    num_users, num_items)
```

#### **Training and Evaluating**
We define an evaluator that uses a trained MF_BPR network and the interactions of users/items to produce negative items and, using these negative items, compute the hitting rate and AUC. 
```python
def evaluate_ranking_bpr(net, test_input, interactions, num_users, num_items):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_items)])
    for u in range(num_users):
        neg_items = list(all_items - set(interactions[u]))
        user_ids, item_ids, scores = [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        test_dataset = data.TensorDataset(torch.from_numpy(np.array(user_ids)),    
            torch.from_numpy(np.array(item_ids)))
        test_data_iter = data.DataLoader(test_dataset, shuffle=False, batch_size=1024)

        for _, (user_idxs, item_idxs) in enumerate(test_data_iter):
            scores.extend(list(net(user_idxs, item_idxs).detach().numpy()))
        item_scores = list(zip(item_ids, scores))

        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        
        temp = metrics.hit_and_auc(ranked_items[u], test_input[u][0], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    return np.mean(np.array(hit_rate)), np.mean(np.array(auc))
```
We are now ready to create and initialize our models with hyperparameters defined by the paper we're trying to replicate as well as standard values in literature for these models.  We use an Adam optimizer with weight decay to optimize our model.
```python
lr, num_epochs, wd, latent_factors = 0.01, 20, 1e-5, 10

bpr_net = mf_bpr.MF_BPR(num_users, num_items, latent_factors) 
loss = mf_bpr.BPR_Loss
optimizer = optim.Adam(bpr_net.parameters(), lr = 0.01, weight_decay=wd)
```
With the model created and initialized, we are ready to train and evaluate:
```python
hit_rate_list_bpr = []
auc_list_bpr = []
for epoch in range(num_epochs):
    accumulator, l = utils.Accumulator(2), 0.

    # Train each batch
    bpr_net.train()
    for i, (user_idxs, item_idxs, neg_items) in enumerate(train_dataloader):
        optimizer.zero_grad()

        p_pos = bpr_net(user_idxs, item_idxs)
        p_neg = bpr_net(user_idxs, neg_items)

        total_loss = loss(p_pos, p_neg)
        total_loss.backward()
        optimizer.step()
        accumulator.add(total_loss, user_idxs.shape[0])

    # Evaluate
    bpr_net.eval()
    hit_rate, auc = evaluate_ranking_bpr(bpr_net, test_interactions, interactions,
        num_users, num_items)
    hit_rate_list_bpr.append(hit_rate)
    auc_list_bpr.append(auc)

    print(f"Epoch {epoch}:\n\tloss = {accumulator[0]/accumulator[1]}\n\thit_rate = {hit_rate}\n\tauc = {auc}")
```
Here are the results of the ```MF_BPR``` model on the MovieLens dataset:
```python
x = list(range(1, num_epochs + 1))
plt.scatter(x, auc_list_bpr, label = "AUC")
plt.scatter(x, hit_rate_list_bpr, label = "Hit Rate")
plt.title("HR and AUC over Epoch of MF")
plt.xlabel("Epoch")
plt.legend(loc = "lower right")
plt.xticks(x[0::2])
plt.ylim((0, 1))
```
<p align="center">
<img src="images/mf_dummy.png" alt = "mf_dummy" width="66%"/>
<br/><br/>
</p>

### **Alternating Least Squares**
#### **Loading Data**
Like in the ```MF_BPR``` case, we define two functions to execute a train/test split as well as load the data into user, item, score, and interactions iterables.
```python
def train_test_dummy_als(dummy_data : pd.DataFrame, num_users : int, num_items : int):
    train_items, test_items, train_list = {}, {}, []

    # Iterate through every line in the raw data
    for line in dummy_data.itertuples():
        u, i, rating, time = line[1], line[2], line[3], line[4]
        train_items.setdefault(u, []).append((u, i, rating, time))
        if u not in test_items or test_items[u][2] < time:
            test_items[u] = (i, rating, time)
        
    # Iterate through every user and add their samples, sorted by timestamp, to the train 
    # list
    for u in range(1, num_users + 1):
        train_list.extend(sorted(train_items[u], key = (lambda x : x[3])))

    test_data = [(key, *value) for key, value in test_items.items()]

    train_data = [item for item in train_list if item not in test_data]
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    return train_data, test_data

def load_dummy_als(dummy, num_users, num_items):
    users, items, scores = [], [], []
    interactions = {}
    for line in dummy.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = 1 # implicit

        users.append(user_index)
        items.append(item_index)
        scores.append(score)

        interactions.setdefault(user_index, []).append(item_index)

    return users, items, scores, interactions
```
Since we're working with analytical solutions, there is no need to create PyTorch ```Dataset``` or ```DataLoader``` objects.
```python
# Ready dummy data
dummy_data, num_users, num_items = read_dummy()
train_dummy, test_dummy = train_test_dummy_als(dummy_data, num_users, num_items)

# Training data
train_users, train_items, train_ratings, interactions = load_dummy_als(train_dummy,    
    num_users, num_items)

# Test data
_, _, _, test_interactions = load_dummy_als(test_dummy, 
    num_users, num_items)
```

#### **Training and Evaluating**
As before, we define an evaluator to use the trained network and user/item interactions to compute hitting rate and AUC values.
```python
def evaluate_ranking_als(net, test_input, interactions, num_users, num_items):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_items)])
    for u in range(num_users):
        neg_items = list(all_items - set(interactions[u]))
        user_ids, item_ids, scores = [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]

        scores.extend(list(net.predict(user_ids, item_ids)))
        item_scores = list(zip(item_ids, scores))

        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        
        temp = metrics.hit_and_auc(ranked_items[u], test_input[u][0], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    return np.mean(np.array(hit_rate)), np.mean(np.array(auc))
```
We can now initialize the model with hyperparameters chosen based on the paper I am replicating.  Due to the analytical solutions, there is no need to use an optimizer.
```python
num_epochs, reg, latent_factors = 20, 0.01, 30

ratings_matrix = coo_matrix((train_ratings, (train_users, train_items)), shape = (num_users, 
    num_items)).todense()
als_net = als.ALS(num_users, num_items, latent_factors, ratings_matrix, reg)
```
Let's train and evaluate, now with a simpler loop due to absence of batching:
```python
hit_rate_list_als = []
auc_list_als = []
 
for epoch in range(num_epochs):
    # Train with entire batch
    als_net.train()

    # Evaluate
    hit_rate, auc = evaluate_ranking_als(als_net, test_interactions, interactions, num_users,
        num_items)
    hit_rate_list_als.append(hit_rate)
    auc_list_als.append(auc)

    print(f"Epoch {epoch}: hit_rate = {hit_rate}, auc = {auc}")
```
Here are the results of the ```ALS``` model on the MovieLens dataset:
```python
x = list(range(1, num_epochs + 1))
plt.scatter(x, auc_list_als, label = "AUC")
plt.scatter(x, hit_rate_list_als, label = "Hit Rate")
plt.title("HR and AUC over Epoch of ALS")
plt.xlabel("Epoch")
plt.legend(loc = "lower right")
plt.xticks(x[0::2])
plt.ylim((0, 1))
```
<p align="center">
<img src="images/als_dummy.png" alt = "als_dummy" width="66%"/>
<br/><br/>
</p>

### **Word2Vec**
#### **Loading Data**
In our Word2Vec model, we don't care about the users.  We only care about the items in each portfolio.  More specifically, given an item and a window size of 10, we care about the previous 10 items as our context.  To enable this process, we must first load the interactions of each user, sorted by time.
```python
def load_interactions_cbow(dummy_data : pd.DataFrame):
    interactions = {}
    for line in dummy_data.itertuples():
        user_index, item_index, time = line[1] - 1, line[2] - 1, line[4]
        interactions.setdefault(user_index, []).append((item_index, time))

    interactions = {k : sorted(v, key = (lambda pair : pair[1])) for k, v in interactions.items()}
    return {k : [x[0] for x in v] for k, v in interactions.items()}
```
Then, we split these interactions into train and test sets, each set containing its own targets and contexts.
```python
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
```
Given these functions, we can now prepare the data and PyTorch ```DataLoader``` object.
```python
window = 10

dummy_data, num_users, num_items = read_dummy()
sorted_interactions = load_interactions_cbow(dummy_data)
train_targets, train_contexts, test_targets, test_contexts = train_test_dummy_cbow(sorted_interactions, window)

ngrams_train = data.TensorDataset(torch.from_numpy(np.array(train_targets)), 
        torch.from_numpy(np.array(train_contexts)))
ngrams_dataloader = data.DataLoader(dataset = ngrams_train, batch_size = 1024, 
    shuffle = True, num_workers = 4)
ngrams_test = data.TensorDataset(torch.from_numpy(np.array(test_targets)), 
    torch.from_numpy(np.array(test_contexts)))
ngrams_dataloader_test = data.DataLoader(dataset = ngrams_test, batch_size = 1024, 
    shuffle = False, num_workers = 4)
```

#### **Training and Evaluating**
The evaluator functions is similar to the ones from before, except now we're just dealing with interactions instead of user ids and item ids.
```python
def evaluate_ranking_cbow(net, test_targets, test_contexts, num_items):
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
```
We are now ready to prepare the model with an embedding_dim of ```30```.  As mentioned during Word2Vec/CBOW's description, we use ```nn.NLLLoss``` from the PyTorch library.  We also use the Adam optimizer, like in the ```MF_BPR``` training regiment.
```python
embedding_dim, num_epochs, learning_rate = 30, 20, 0.025
loss = torch.nn.NLLLoss()
cbow_net = word2vec.CBOW(num_items, embedding_dim, window)
optimizer = optim.Adam(cbow_net.parameters(), lr = learning_rate)
```
Given this model, we can now train and evaluate.
```python
hit_rate_list_cbow = []
auc_list_cbow = []
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
    hit_rate, auc = evaluate_ranking_cbow(cbow_net, test_targets, test_contexts, num_items)
    hit_rate_list_cbow.append(hit_rate)
    auc_list_cbow.append(auc)

    print(f"Epoch {epoch}:\n\tloss = {accumulator[0]/accumulator[1]}\n\thit_rate = {hit_rate}\n\tauc = {auc}")
```
Here are the results of the ```CBOW``` model on the MovieLens dataset:
```python
x = list(range(1, num_epochs + 1))
plt.scatter(x, auc_list_cbow, label = "AUC")
plt.scatter(x, hit_rate_list_cbow, label = "Hit Rate")
plt.title("HR and AUC over Epoch of CBOW")
plt.xlabel("Epoch")
plt.legend(loc = "lower right")
plt.xticks(x[0::2])
plt.ylim((0, 1))
```
<p align="center">
<img src="images/cbow_dummy.png" alt = "cbow_dummy" width="66%"/>
<br/><br/>
</p>

For all three models, we can clearly see that the AUC and hitting rate values are good, thereby implying that our models are indeed effective in this dummy case.  Let's now move on to more interesting data.
<br/><br/>

## Using the Model on Representative Data
To start, let's read in the [dataset](data/data_UCI.csv):