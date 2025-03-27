import torch

import torch.nn as nn

from torch.utils.data import Dataset


class SequenceDatasetOH(Dataset):
    def __init__(self, data):
        self.X = data[:, 0]
        self.y = data[:, 1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_tensor, y_tensor
    

class SequenceDatasetEmbedding(Dataset):
    def __init__(self, data):
        self.X = data[:, :-1]
        self.y = data[:, -1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.long)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return X_tensor, y_tensor


class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLayer, self).__init__()
        self.weight = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
            
    def forward(self, x):
        output = torch.matmul(x, self.weight.t()) + self.bias
        return output


class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CustomEmbedding, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        nn.init.xavier_uniform_(self.embedding_matrix)
    
    def forward(self, indices):
        return self.embedding_matrix[indices]
    

class CustomNNOH(nn.Module):
    def __init__(self):
        super(CustomNNOH, self).__init__()
        self.layer1 = CustomLayer(50940, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.activation1 = nn.Sigmoid()
        
        self.layer2 = CustomLayer(2048, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.activation2 = nn.Sigmoid()
        
        self.layerF = CustomLayer(256, 19)
        self.activationF = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        
        x = self.layerF(x)
        x = self.activationF(x)
        return x
    

class CustomNNEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=42, seq_len=20):
        super(CustomNNEmbedding, self).__init__()

        self.embedding = CustomEmbedding(vocab_size, embedding_dim)

        self.layer1 = CustomLayer(seq_len * embedding_dim, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.activation1 = nn.Sigmoid()
        
        self.layer2 = CustomLayer(1000, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.activation2 = nn.Sigmoid()
        
        self.layerF = CustomLayer(200, 19)
        self.activationF = nn.Softmax(dim=1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        
        batch_size = embedded.size(0)
        embedded_flat = embedded.view(batch_size, -1)
        x = self.layer1(embedded_flat)
        x = self.bn1(x)
        x = self.activation1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        
        x = self.layerF(x)
        x = self.activationF(x)
        return x
