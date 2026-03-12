import torch
import torch.nn as nn

class ClientLSTM(nn.Module):
    """
    The local model deployed on the Edge/IoT device. 
    It is responsible for taking raw D-dimensional weather features over a time window
    and extracting abstract 'smashed activations' (hidden states). 
    """
    def __init__(self, input_size=5, hidden_size=64, num_layers=1):
        super(ClientLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Core feature extractor 
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_size) 
        output shape: (batch_size, hidden_size) - the Smashed Activation
        """
        # lstm returns: output, (h_n, c_n)
        # We only care about the final hidden state of the sequence to send to the server
        _, (h_n, _) = self.lstm(x)
        
        # h_n shape: (num_layers, batch_size, hidden_size)
        # We take the output of the final layer and return shape: (batch_size, hidden_size)
        smashed_activation = h_n[-1]
        
        return smashed_activation


class ServerHead(nn.Module):
    """
    The central model residing on the cloud server. 
    It takes the abstract 'smashed activations' from the clients over the network
    and finishes the predictive calculation.
    """
    def __init__(self, hidden_size=64, output_size=1):
        super(ServerHead, self).__init__()
        
        # Back to a more robust MLP structure
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LeakyReLU(0.1), # Allows gradients to flow even when input < 0
            nn.Linear(32, output_size),
            nn.ReLU()          # Rain is >= 0
        )
        
    def forward(self, smashed_activation):
        """
        smashed_activation shape: (batch_size, hidden_size)
        output shape: (batch_size, output_size) -> predicted rainfall amount
        """
        prediction = self.regressor(smashed_activation)
        return prediction
