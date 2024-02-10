import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
seed_value = 42 #set the seed value
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10
hidden_size = 128
batch_size = 32
epochs = 10

eng_prefixes = (
"i am ", "i m ",
"he is", "he s ",
"she is", "she s ",
"you are", "you re ",
"we are", "we re ",
"they are", "they re "
)
