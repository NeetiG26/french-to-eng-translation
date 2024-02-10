import torch
import random
import matplotlib.pyplot as plt
from src.features.build_features import get_dataloader
from src.models.attention_decoder import AttnDecoderRNN
from src.models.encoder import EncoderRNN
from src.models.predict_model import evaluateRandomly
from src.models.train_model import train
from src.visualization.visualize import evaluateAndShowAttention
from config import seed_value, batch_size, hidden_size, epochs, device
import warnings
warnings.filterwarnings("ignore")


plt.switch_backend('agg')

print("device :", device)



torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
random.seed(seed_value)


print("calculation dataloader")
input_lang, output_lang, train_dataloader = get_dataloader(batch_size)
print("Dataloader Done")
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
print("Encoder :", encoder)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
print("Decoder :", decoder)

train(train_dataloader, encoder, decoder, epochs, print_every=1, plot_every=5)
encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder)

evaluateAndShowAttention('il n est pas aussi grand que son pere')
evaluateAndShowAttention('je suis reellement fiere de vous')
