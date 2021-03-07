import torch
import torch.nn as nn


class ALS(nn.Module):
    def __init__(self, investor_num: int, stock_num: int, latent_factors: int):
        """
        An Alternating Least Squares model which alternates between freezing the
        investor embedding matrix and the stock embedding matrix

        Parameters
        ----------
        investor_num (int) - number of investors\n
        stock_num (int) - number of stocks\n
        latent_factors (int) - number of latent factors
        """
        super(ALS, self).__init__()

        self.embed_investor = nn.Embedding(investor_num, latent_factors)
        self.embed_stock = nn.Embedding(stock_num, latent_factors)

        nn.init.normal_(self.embed_investor.weight, std=0.01)
        nn.init.normal_(self.embed_stock.weight, std=0.01)

    def forward(self, investor: torch.Tensor, stock_positive: torch.Tensor,
                investor_train: bool) -> torch.Tensor:
        """
        Parameters
        ----------
        investor (torch.Tensor) - investor ids\n
        stock_positive (torch.Tensor) - ids of stocks that investor purchased\n
        investor_train (bool) - flag that indicates if we want to train the investor
            embeddings and keep the stock embeddings constant

        Output
        ------
        prediction_positive (torch.Tensor) - current prediction of a stock that the
         investor purchased
        """
        self.embed_investor.requires_grad_(investor_train)
        self.embed_stock.requires_grad_(not investor_train)

        investor = self.embed_investor(investor)
        stock_positive = self.embed_stock(stock_positive)

        prediction_positive = (investor * stock_positive).sum(dim=-1)
        return prediction_positive
