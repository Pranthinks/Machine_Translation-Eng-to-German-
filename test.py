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
from transformer import FullTransformer_Custom

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
#Adding Validation Step after each loop
vocab_size = max(len(eng_tokens), len(ger_tokens))
model = FullTransformer_Custom(vocab_size, heads = 8, d_model=512, hidden_lay=1024, seq_len=100, num_layers=2)
model = model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=ger_tokens["<pad>"])
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
epochs = 10
scaler = torch.cuda.amp.GradScaler()


for i in range(epochs):
    print("Running Epoch Number", i)
    model.train()
    total_loss = 0
    batch_count = 0
    for src, tgr, _, _ in train_loader:
        src = src.to(device)
        tgr = tgr.to(device)

        tgr_input = tgr[:, :-1]
        tgr_output = tgr[:, 1:]
        optimizer.zero_grad()
   
        #Adding Mixed Precesion to speed up the training
        with torch.cuda.amp.autocast():
             output = model(src, tgr_input)
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
         for src, tgr, _, _ in val_loader:
             src = src.to(device)
             tgr = tgr.to(device)

             tgr_input = tgr[:, :-1]
             tgr_output = tgr[:, 1:]
        
             #Adding Mixed Precesion to speed up the training
             with torch.cuda.amp.autocast():
                  output = model(src, tgr_input)
                  loss = criterion(output.reshape(-1, vocab_size), tgr_output.reshape(-1))
        
             val_loss += loss.item()
             val_count += 1
    avg_val_loss = val_loss / val_count
    print(f"Epoch {i+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Perplexity: {math.exp(avg_val_loss):.2f}\n")
