import torch
import torch.nn as nn

class ClientLSTM(nn.Module):
    """
    The local model deployed on the Edge/IoT device. 
    It is responsible for taking raw D-dimensional weather features over a time window
    and extracting abstract 'smashed activations' (hidden states). 
    """
    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        num_layers=1,
        lstm_dropout=0.3,
        dropout=None,
    ):
        super(ClientLSTM, self).__init__()
        if dropout is not None:
            # Backward-compatible alias for legacy call sites.
            lstm_dropout = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_dropout = float(lstm_dropout)
        
        # Core feature extractor 
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=self.lstm_dropout if num_layers > 1 else 0.0
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
    def __init__(self, hidden_size=64, output_size=1, head_width=64, dropout=0.1):
        super(ServerHead, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(hidden_size, head_width),
            nn.LayerNorm(head_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, head_width),
            nn.SiLU(),
        )
        self.rain_classifier = nn.Linear(head_width, 1)
        self.rain_regressor = nn.Linear(head_width, output_size)

    def forward(self, smashed_activation):
        """
        smashed_activation shape: (batch_size, hidden_size)
        returns:
          rain_logit: unnormalized rain/no-rain score
          rain_amount: predicted rainfall amount in transformed space
        """
        features = self.backbone(smashed_activation)
        rain_logit = self.rain_classifier(features)
        rain_amount = self.rain_regressor(features)
        return rain_logit, rain_amount
