import torch
import torch.nn as nn

class LoanDefaultClassifier(nn.Module):
   
    def __init__(self, input_features, hidden_layers, dropout_rate=0.3):
        super(LoanDefaultClassifier, self).__init__()
        layers = []
        # Input layer
        in_features = input_features
        
        # Hidden layers
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
            
        # Output layer
        # A single output neuron for binary classification
        layers.append(nn.Linear(in_features, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)