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

# Loading the Data
train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=('en', 'de'))

#Initializing the Tokeniziers
tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")
tokenizer_de = get_tokenizer("spacy", language="de_core_news_sm")

#Method to tokenizer on each word of the sentence
def yeild_tokens(datasets, tokenizer, index = 0):
    for data in datasets:
        yield tokenizer(data[index])

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
train_loader = DataLoader(train_iter, batch_size = 32, collate_fn=collate_fn)
val_loader = DataLoader(valid_iter, batch_size = 32, collate_fn=collate_fn)
test_loader = DataLoader(test_iter, batch_size = 32, collate_fn=collate_fn)

print("\nTesting DataLoader...")
src, tgt, src_mask, tgt_mask = next(iter(train_loader))
print(f"Source shape: {src.shape}")
print(f"Target shape: {tgt.shape}")
print(f"Source mask shape: {src_mask.shape}")
print(f"Target mask shape: {tgt_mask.shape}")
