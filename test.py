'''
Willing to write the Data Loading part which gonna take care of all the
preprocessing steps that has to be taken before feding the data to the Transformer
'''
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from torchtext.datasets import Multi30k
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
from itertools import islice
from transformer import *

# Loading the Data
train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=('en', 'de'))

#Initializing the Tokeniziers
tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")
tokenizer_de = get_tokenizer("spacy", language="de_core_news_sm")

#Method to tokenizer on each word of the sentence
def yeild_tokens(datasets, tokenizer, index = 0):
    for data in datasets:
        yield tokenizer(data[index])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Using default build_vocab_from_iterator to find all the unique words and assigning them an integer
#For english sentences
eng_tokens = build_vocab_from_iterator(yeild_tokens(train_iter, tokenizer_en), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
eng_tokens.set_default_index(eng_tokens["<unk>"])

train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=('en', 'de'))
#For German Sentences so keeping the index as 1
ger_tokens = build_vocab_from_iterator(yeild_tokens(train_iter, tokenizer_de, index=1), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
ger_tokens.set_default_index(ger_tokens["<unk>"])
#print(len(eng_tokens))
#print(len(ger_tokens))

#Method to apply these vocab to data
def apply_vocab(src_text, tgr_text):
    src_tokens = [eng_tokens["<bos>"]] + [eng_tokens[t] for t in tokenizer_en(src_text)] + [eng_tokens["<eos>"]]
    tgt_tokens = [ger_tokens["<bos>"]] + [ger_tokens[t] for t in tokenizer_de(tgr_text)] + [ger_tokens["<eos>"]]

    src_tensor = torch.tensor(src_tokens)
    tgr_tensor = torch.tensor(tgt_tokens)

    return src_tensor, tgr_tensor

#Method to apply padding
def padding(src, tar):
    pad_src = pad_sequence(src, batch_first= True, padding_value=eng_tokens["<pad>"])
    pad_tar = pad_sequence(tar, batch_first=True, padding_value=ger_tokens["<pad>"])

    src_op = (pad_src == eng_tokens["<pad>"])
    tar_op = (pad_tar == ger_tokens["<pad>"])

    return pad_src, pad_tar, src_op, tar_op

#Applying custom apply_vocab and padding function to the train data using a function called collat
def collate_fn(batch):
    src_list , tgr_list = [], []

    for src, tgr in batch:
        src_tensor , tgr_tensor = apply_vocab(src, tgr)
        src_list.append(src_tensor)
        tgr_list.append(tgr_tensor)

    return padding(src_list, tgr_list)

#Now we has to write a Custom Dataloader that uses this collate function
train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=('en', 'de'))
train_loader = DataLoader(train_iter, batch_size = 8, collate_fn=collate_fn)
val_loader = DataLoader(valid_iter, batch_size = 8, collate_fn=collate_fn)
test_loader = DataLoader(test_iter, batch_size = 8, collate_fn=collate_fn)

print("\nTesting DataLoader...")
src, tgt, src_mask, tgt_mask = next(iter(train_loader))
print(f"Source shape: {src.shape}")
print(f"Target shape: {tgt.shape}")
print(f"Source mask shape: {src_mask.shape}")
print(f"Target mask shape: {tgt_mask.shape}")


'''
Training the Custom Transformer model with Mixied precesion and Gradient Clipping
'''
from torch.optim.lr_scheduler import ReduceLROnPlateau

#Adding Validation Step after each loop
vocab_size = max(len(eng_tokens), len(ger_tokens))
model = FullTransformer_Custom(vocab_size, heads = 8, d_model=512, hidden_lay=1024, seq_len=100, num_layers=2)
model = model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=ger_tokens["<pad>"])
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
epochs = 5
scaler = torch.cuda.amp.GradScaler()


for i in range(epochs):
    print("Running Epoch Number", i)
    model.train()
    total_loss = 0
    batch_count = 0
    for src, tgr, src_mask, tgr_mask in train_loader:
        src = src.to(device)
        tgr = tgr.to(device)
        src_mask = src_mask.to(device)
        tgr_mask = tgr_mask.to(device)

        tgr_input = tgr[:, :-1]
        tgr_mask_input = tgr_mask[:, :-1]
        tgr_output = tgr[:, 1:]
        optimizer.zero_grad()

        #Adding Mixed Precesion to speed up the training
        with torch.cuda.amp.autocast():
             output = model(src, tgr_input, src_mask=src_mask, tgt_mask=tgr_mask_input)
             loss = criterion(output.reshape(-1, vocab_size), tgr_output.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_count += 1
    avg_loss = total_loss / batch_count
    print(f"Epoch {i+1}/{epochs}, Loss: {avg_loss:.4f}, Perplexity: {math.exp(avg_loss):.2f}")

    #Adding Validation after each Epoch to generalize the model better
    model.eval()
    val_loss = 0
    val_count = 0
    with torch.no_grad():
         for src, tgr, src_mask, tgr_mask in val_loader:
             src = src.to(device)
             tgr = tgr.to(device)
             src_mask = src_mask.to(device)
             tgr_mask = tgr_mask.to(device)
             tgr_input = tgr[:, :-1]
             tgr_output = tgr[:, 1:]
             tgr_mask_input = tgr_mask[:, :-1]

             #Adding Mixed Precesion to speed up the training
             with torch.cuda.amp.autocast():
                  output = model(src, tgr_input, src_mask=src_mask, tgt_mask=tgr_mask_input)
                  loss = criterion(output.reshape(-1, vocab_size), tgr_output.reshape(-1))

             val_loss += loss.item()
             val_count += 1
    avg_val_loss = val_loss / val_count
    print(f"Epoch {i+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Perplexity: {math.exp(avg_val_loss):.2f}\n")
    scheduler.step(avg_val_loss)


#Saving the Model
# Save the model after training completes
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': avg_loss,
    'val_loss': avg_val_loss,
}, 'transformer_model.pth')

print("Model saved successfully!")


#TESTING THE TRANSFORMER ON TEST DATA

import urllib.request
import gzip
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# Download and prepare test data
def download_multi30k_test():
    """Download Multi30k test data, bypassing corrupted cache"""
    data_dir = Path('/tmp/multi30k_manual')
    data_dir.mkdir(exist_ok=True)

    base_url = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    files = {'en': 'test_2016_flickr.en.gz', 'de': 'test_2016_flickr.de.gz'}

    data = {}
    for lang, filename in files.items():
        txt_path = data_dir / f'test_{lang}.txt'

        if not txt_path.exists():
            urllib.request.urlretrieve(base_url + filename, data_dir / filename)
            with gzip.open(data_dir / filename, 'rb') as f_in, open(txt_path, 'wb') as f_out:
                f_out.write(f_in.read())
            (data_dir / filename).unlink()  # Remove .gz

        with open(txt_path, 'r', encoding='utf-8') as f:
            data[lang] = [line.strip() for line in f]

    return data['en'], data['de']

# Simple dataset class
class SimpleDataset(Dataset):
    def __init__(self, src, tgt):
        self.data = list(zip(src, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Download and test
test_en, test_de = download_multi30k_test()
test_loader = DataLoader(SimpleDataset(test_en, test_de), batch_size=8, collate_fn=collate_fn)

#Loading the Saved Model
checkpoint = torch.load('transformer_model.pth', map_location=device)  # Changed to match your save name
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Model loaded from epoch {checkpoint['epoch']}")
print(f"Saved Val Loss: {checkpoint['val_loss']:.4f}\n")

# Evaluate
model.eval()
test_loss = 0
with torch.no_grad():
    for src, tgt, src_mask, tgt_mask in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

        tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

        with torch.cuda.amp.autocast():
            output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask[:, :-1])
            test_loss += criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1)).item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f} | Perplexity: {math.exp(avg_test_loss):.2f}")