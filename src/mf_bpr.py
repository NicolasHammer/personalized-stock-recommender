import torch
import torch.nn as nn


def BPR_Loss(positive, negative):
    distances = positive - negative
    loss = - torch.sum(torch.log(torch.sigmoid(distances)), 0, keepdim=True)

    return loss


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

    def forward(self, investor_num: torch.Tensor, stock_num: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        investor (torch.Tensor) - investor id\n
        stock (torch.Tensor) - ids of stocks that investors purchased\n

        Output
        ------
        prediction  (torch.Tensor) - current prediction of stocks that the
        investors purchased
        """
        investor = self.embed_investor(investor_num)
        stock_positive = self.embed_stock(stock_num)
        prediction = (investor * stock_positive).sum(dim=-1)

        return prediction
