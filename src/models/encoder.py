import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # Initialize an embedding layer for converting input indexes to dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Create a Gated Recurrent Unit (GRU) layer with the specified hidden size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        # Pass the embedded vectors through the GRU layer
        # 'output' contains the GRU outputs for each time step in the sequence
        # 'hidden' is the final hidden state of the GRU after processing the sequence
        output, hidden = self.gru(embedded)
        return output, hidden