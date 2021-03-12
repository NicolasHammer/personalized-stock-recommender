import numpy as np


class ALS():
    def __init__(self, investor_num: int, stock_num: int, latent_factors: int, train_data: np.ndarray,
                 reg: float):
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
