#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from tqdm import tqdm
import matplotlib.pyplot as plt

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

file, file_len = read_file(args.filename)

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

def make_plot(perplexity):
    plt.figure(0)
    plt.plot(perplexity)
    plt.title("change in perplexity during training")
    plt.ylabel('perplexity')
    plt.xlabel('epoch')
    plt.figure(1)
    plt.plot(perplexity)
    plt.title("change in perplexity during training")
    plt.ylabel('perplexity')
    plt.xlabel('epoch')
    plt.ylim(0, 10)
    plt.show()

def print_strings():
    print("Generating Strings", '\n')
    print("Prime: iwncj", '\n')
    print(generate(decoder, 'iwncj', 100, cuda=args.cuda), '\n')
    print("Prime: to5 p", '\n')
    print(generate(decoder, 'to5 p', 100, cuda=args.cuda), '\n')
    print("Prime: vvmnw", '\n')
    print(generate(decoder, 'vvmnw', 100, cuda=args.cuda), '\n')

    print("Prime: The", '\n')
    print(generate(decoder, 'The', 100, cuda=args.cuda), '\n')
    print("Prime: What is", '\n')
    print(generate(decoder, 'What is', 100, cuda=args.cuda), '\n')
    print("Prime: Shall I give", '\n')
    print(generate(decoder, 'Shall I give', 100, cuda=args.cuda), '\n')
    print("Prime: X087hNYB BHN BYFVuhsdbs", '\n')
    print(generate(decoder, 'X087hNYB BHN BYFVuhsdbs', 100, cuda=args.cuda), '\n')


# Initialize models and start training

decoder = CharRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0
perplexity = []


try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.chunk_len, args.batch_size))
        loss_avg += loss

        perplexity.append(math.exp(loss))

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

    print_strings()

    print("Saving...")
    save()
    make_plot(perplexity)

except KeyboardInterrupt:
    print_strings()
    print("Saving before quit...")
    save()
    make_plot(perplexity)

