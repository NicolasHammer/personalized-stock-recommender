import torch
import torch.nn as nn
import torch.functional as F

class CBOW(nn.Module):
    def __init__(self, vocab : list, embedding_dim : int, context_size : int):
        """
        Initialize a CBOW model. Adapted from https://srijithr.gitlab.io/post/word2vec/.

        Parameters
        ----------
        vocab (list) - a list of every stock under consideration\n
        embedding_dim (int) - the size of the embedding dimension\n
        context_size (int) - the number of words used as context for prediction
        """
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, len(vocab))
        self.word_to_ix = {word : i for i, word in enumerate(vocab)}

    def forward(self, context_words : list) -> torch.Tensor:
        """
        Parameters
        ----------
        context_words (list) - a list of words ids used as context for prediction of
            the target

        Output
        ------
        log_probabilities (torch.Tensor) - a tensor of the probabilities of each word
            in the vocabulary
        """
        embeddings = self.embeddings(context_words).view((1, -1)) 
        out1 = F.relu(self.linear1(embeddings)) 
        out2 = self.linear2(out1)           
        log_probabilities = F.log_softmax(out2, dim=1)
        return log_probabilities

    def predict(self, input) -> None    :
        """
        Parameters
        ----------
        input (list) - a list of words used as context for prediction

        Output
        ------
        A print out of the stocks with highest probabilities to be present
        """
        context_idxs = torch.tensor([self.word_to_ix[w] for w in input], dtype=torch.long)
        res = self.forward(context_idxs)
        res_arg = torch.argmax(res)
        res_val, res_ind = res.sort(descending=True)
        res_val = res_val[0][:3]
        res_ind = res_ind[0][:3]
        for arg in zip(res_val,res_ind):
            print([(key,val,arg[0]) for key,val in self.word_to_ix.items() if val == arg[1]])

    def freeze_layer(self,layer):
        for name,child in model.named_children():
            if(name == layer):
                for names,params in child.named_parameters():
                    print(names,params)
                    print(params.size())
                    params.requires_grad= False