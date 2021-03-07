import torch
import torch.nn as nn

import numpy as np


class MF_BPR(nn.Module):
    def __init__(self, investor_num: int, stock_num: int, latent_factors: int):
        """
        Initializes a matrix factorization model that is meant to be used in conjunction
        with Bayesian Personalized Recommendation loss.

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

    def forward(self, investor: torch.Tensor, stock: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        investor (torch.Tensor) - investor ids\n
        stock_positive (torch.Tensor) - ids of stocks that investors purchased\n

        Output
        ------
        prediction_positive (torch.Tensor) - current prediction of stocks that the
         investors purchased\n
        """
        investor = self.embed_investor(investor)
        stock = self.embed_stock(stock)
        prediction = (investor * stock).sum(dim=-1)

        return prediction
