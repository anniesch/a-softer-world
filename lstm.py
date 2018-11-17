import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SofterLSTM(nn.Module):
    def __init__(self, ):
        super(SofterLSTM, self).__init__()

    self.embedding = nn.Embedding(vocab_dim, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim)
