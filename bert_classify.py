import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from transformers import WarmupLinearSchedule, BertForSequenceClassification, BertTokenizer, BertConfig

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

def prepare_dataset(text, label, msl, tokenizer):
    X_set = []
    for line in tqdm(text):
        tokens = tokenizer.tokenize(line)
        x = tokens[:-1][:msl - 1] + ['[CLS]']
    
        if len(x) < msl:
            x = x + ['[PAD]' for _ in range(msl - len(x))]
        x = tokenizer.convert_tokens_to_ids(x) 
        X_set.append(x)
    
    X_set = torch.LongTensor(X_set)
    y_set = torch.LongTensor(label)
    data = data_utils.TensorDataset(X_set, y_set)
  
    return data

def train(model, criterion, optimizer, train_loader, max_norm=0.25, scheduler=None):
    train_loss = 0
    train_acc = 0

    model.train()
    for batch in tqdm(train_loader):
        x, y = batch

        x = x.to(device)
        y = y.to(device)

        logits = model(x)[0]
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        train_loss += loss.item()
        train_acc += torch.sum(torch.argmax(logits, dim=1) == y).item() / len(y)
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
    return train_loss, train_acc

def validate(model, criterion, valid_loader):
    valid_loss = 0
    valid_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            x, y = batch

            x = x.to(device)
            y = y.to(device)

            logits = model(x)[0]
            loss = criterion(logits, y)

            valid_loss += loss.item()
            valid_acc += torch.sum(torch.argmax(logits, dim=1) == y).item() / len(y)
        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader)
    
    return valid_loss, valid_acc

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None, type=str, required=True, help="Data location, must be file.csv")
parser.add_argument("--output", default=None, type=str, required=True, help="Output location, full filename")
parser.add_argument("--filename", default="train", help="Filename of dataset")
parser.add_argument("--train_size", default=0.7, type=float, help="size of training split")
parser.add_argument("--config", default="bert_config.json", type=str, help="Config file in JSON")
parser.add_argument("--model", default=None, type=str, required=True, help="Directory of saved model")

parser.add_argument("--epochs", default=3, type=int, help="Number of training epochs")
parser.add_argument("--msl", default=512, type=int, help="Maximum sequence length")
parser.add_argument("--bs", default=32, type=int, help="Batch size")
parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")
parser.add_argument("--warmup", default=0.1, type=float, help="Percentage of steps to warmup the learning rate")
parser.add_argument("--max_norm", default=1.0, type=float, help="Gradient clipping")

parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--cuda", action='store_true', help='Use CUDA')
parser.add_argument("--multi", action='store_true', help='Use multiple GPU')
parser.add_argument("--gpus", default=4, type=int, help="Number of GPUs to use")
args = parser.parse_args()

print(args)

# Set device and random seeds
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
torch.manual_seed(args.seed);
torch.cuda.manual_seed(args.seed);
torch.backends.cudnn.deterministic = True

# Prepare Dataset
cachename = '_'.join([args.filename, str(args.bs), str(args.msl), str(args.seed)]) + '.pt'
dr = os.listdir(os.path.expanduser(args.data))
if cachename in dr:
    print("Cached data found. Loading.")
    with open(args.data + '/' + cachename, 'rb') as f:
        train_set, valid_set, train_loader, valid_loader = torch.load(f)
else:
    print("No cached data found. Building cache.")
    df = pd.read_csv(args.data + '/' + args.filename + '.csv').sample(frac=1, random_state=args.seed)
    text, sentiment = list(df['text']), list(df['sentiment'])
    tr_sz = int(len(text) * args.train_size)
    X_train, y_train = text[:tr_sz], sentiment[:tr_sz]
    X_valid, y_valid = text[tr_sz:], sentiment[tr_sz:]
    
    print("Loading tokenizer")
    tokenizer = BertTokenizer.from_pretrained(args.model)

    train_set = prepare_dataset(X_train, y_train, args.msl, tokenizer)
    valid_set = prepare_dataset(X_valid, y_valid, args.msl, tokenizer)
    train_loader = data_utils.DataLoader(train_set, args.bs)
    valid_loader = data_utils.DataLoader(valid_set, args.bs)

    print('Saving cached data')
    with open(args.data + '/' + cachename, 'wb') as f:
        torch.save([train_set, valid_set, train_loader, valid_loader], f)
        
# Prepare Model
print("Loading model and and training setup")
config = BertConfig.from_pretrained(args.model + '/' + args.config)
model = BertForSequenceClassification.from_pretrained(args.model, config=config)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Scheduler
steps = len(train_loader) * args.epochs
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(steps * args.warmup), t_total=steps)

# CUDA and multiple GPUs
if args.multi:
    gpus = [i for i in range(args.gpus)]
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model, device_ids=gpus)
model = model.to(device)

# Train!
print("Beginning training")
for e in range(1, args.epochs + 1):
    train_loss, train_acc = train(model, criterion, optimizer, train_loader, max_norm=args.max_norm, scheduler=scheduler)
    valid_loss, valid_acc = validate(model, criterion, valid_loader)

    print("Epoch {:3} | Train Loss {:.4f} | Train Acc {:.4f} | Valid Loss {:.4f} | Valid Acc {:.4f}".format(e, train_loss, train_acc, valid_loss, valid_acc))
    
# Save model
print("Saving model")
with open(args.output + '.pt', 'wb') as f:
    torch.save(model, f)
print("Done")
