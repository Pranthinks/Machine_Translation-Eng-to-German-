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
from Custom_Transformer import *

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
eng_tokens = build_vocab_from_iterator(yeild_tokens(train_iter, tokenizer_en), specials=["<unk>", "<pad>", "<bos>", "<eos>"], max_tokens=8000)
eng_tokens.set_default_index(eng_tokens["<unk>"])

train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=('en', 'de'))
#For German Sentences so keeping the index as 1
ger_tokens = build_vocab_from_iterator(yeild_tokens(train_iter, tokenizer_de, index=1), specials=["<unk>", "<pad>", "<bos>", "<eos>"], max_tokens=8000)
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



'''
Training the Custom Transformer model with Mixied precesion and Gradient Clipping
'''
from torch.optim.lr_scheduler import ReduceLROnPlateau

#Adding Validation Step after each loop
vocab_size = max(len(eng_tokens), len(ger_tokens))
model = FullTransformer_Custom(vocab_size, heads = 8, d_model=256, hidden_lay=512, seq_len=100, num_layers=3, dropout=0.2)
model = model.to(device)
criterion = nn.CrossEntropyLoss(
    ignore_index=ger_tokens["<pad>"],
    label_smoothing=0.1
)
optimizer = optim.AdamW(  # Changed from Adam
    model.parameters(),
    lr=0.0001,           # Reduced from 0.0003
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.01
)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,          # Reduced from 5
    min_lr=1e-6
)
epochs = 30
scaler = torch.cuda.amp.GradScaler()
best_val_loss = float('inf')

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': i + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
        }, 'best_transformer_model.pth')
        print(f"✓ Best model saved! Val Loss: {avg_val_loss:.4f}\n")


#Saving the Model
# Save the model after training completes
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': avg_loss,
    'val_loss': avg_val_loss,
}, 'final_transformer_model.pth')
print(f"Training complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final model saved as 'final_transformer_model.pth'")
print(f"Best model saved as 'best_transformer_model.pth'")


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
            print(f"Downloading {lang} test data...")
            urllib.request.urlretrieve(base_url + filename, data_dir / filename)
            with gzip.open(data_dir / filename, 'rb') as f_in, open(txt_path, 'wb') as f_out:
                f_out.write(f_in.read())
            (data_dir / filename).unlink()

        with open(txt_path, 'r', encoding='utf-8') as f:
            data[lang] = [line.strip() for line in f]

    return data['en'], data['de']

class SimpleDataset(Dataset):
    def __init__(self, src, tgt):
        self.data = list(zip(src, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Download test data
print("Downloading test data...")
test_en, test_de = download_multi30k_test()
print(f"Test set size: {len(test_en)} examples\n")

test_loader = DataLoader(
    SimpleDataset(test_en, test_de),
    batch_size=8,
    collate_fn=collate_fn,
    shuffle=False
)

# Recreate model with the EXACT architecture that was saved
print("Recreating model to match checkpoint...")
vocab_size = max(len(eng_tokens), len(ger_tokens))
model = FullTransformer_Custom(
    vocab_size,
    heads=8,
    d_model=256,
    hidden_lay=512,  # Your checkpoint was trained with 1024, NOT 2048
    seq_len=100,
    num_layers=3
)
model = model.to(device)

# Load the best model weights
print("Loading best model weights...")
checkpoint = torch.load('best_transformer_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
print(f"✓ Best Val Loss: {checkpoint['val_loss']:.4f}\n")

# Evaluate on test set
print("Evaluating on test set...")
model.eval()
test_loss = 0
batch_count = 0

with torch.no_grad():
    for src, tgt, src_mask, tgt_mask in test_loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask_input = tgt_mask[:, :-1]

        with torch.cuda.amp.autocast():
            output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask_input)
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))

        test_loss += loss.item()
        batch_count += 1

avg_test_loss = test_loss / batch_count
test_perplexity = math.exp(avg_test_loss)

print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)
print(f"Test Loss:       {avg_test_loss:.4f}")
print(f"Test Perplexity: {test_perplexity:.2f}")
print(f"Batches:         {batch_count}")
print("="*50)


def translate_sentence_greedy(sentence, model, eng_tokens, ger_tokens, device, max_len=100):
    """Simple greedy decoding with CORRECT mask format"""
    model.eval()

    # Get vocab mappings
    eng_stoi = eng_tokens.get_stoi()
    ger_stoi = ger_tokens.get_stoi()
    ger_itos = ger_tokens.get_itos()

    # Tokenize source
    tokens = tokenizer_en(sentence.lower())
    src_indices = [eng_stoi["<bos>"]] + [eng_stoi.get(t, eng_stoi["<unk>"]) for t in tokens] + [eng_stoi["<eos>"]]

    # Pad source
    src_len = len(src_indices)
    if len(src_indices) < max_len:
        src_indices += [eng_stoi["<pad>"]] * (max_len - len(src_indices))
    else:
        src_indices = src_indices[:max_len]

    src = torch.tensor([src_indices]).to(device)

    # FIXED: Source mask - True for PAD tokens (to mask them out)
    src_mask = (src == eng_stoi["<pad>"])  # Shape: [batch, seq_len]

    # Start with <bos>
    tgt_indices = [ger_stoi["<bos>"]]

    with torch.no_grad():
        for _ in range(max_len - 1):
            seq_len = len(tgt_indices)

            # Pad current target
            tgt_padded = tgt_indices + [ger_stoi["<pad>"]] * (max_len - seq_len)
            tgt = torch.tensor([tgt_padded]).to(device)

            # FIXED: Target mask - True for PAD tokens
            tgt_mask = (tgt == ger_stoi["<pad>"])  # Shape: [batch, seq_len]

            with torch.cuda.amp.autocast():
                output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

            # Get next token from the LAST GENERATED position
            next_token = output[0, seq_len - 1].argmax().item()

            if next_token == ger_stoi["<eos>"]:
                break

            tgt_indices.append(next_token)

    # Convert to words
    translated_tokens = []
    for idx in tgt_indices[1:]:  # Skip <bos>
        if idx == ger_stoi["<eos>"]:
            break
        token = ger_itos[idx]
        if token not in ["<bos>", "<eos>", "<pad>", "<unk>"]:
            translated_tokens.append(token)

    return ' '.join(translated_tokens) if translated_tokens else "<empty>"


def translate_sentence_beam(sentence, model, eng_tokens, ger_tokens, device, max_len=100, beam_width=5):
    """Translate using beam search with CORRECT mask format"""
    model.eval()

    # Tokenize source
    tokens = tokenizer_en(sentence.lower())

    # Get vocab mappings
    eng_stoi = eng_tokens.get_stoi()
    ger_stoi = ger_tokens.get_stoi()

    src_indices = [eng_stoi["<bos>"]] + [eng_stoi.get(t, eng_stoi["<unk>"]) for t in tokens] + [eng_stoi["<eos>"]]

    # Pad source to max_len
    if len(src_indices) < max_len:
        src_indices += [eng_stoi["<pad>"]] * (max_len - len(src_indices))
    else:
        src_indices = src_indices[:max_len]

    src = torch.tensor([src_indices]).to(device)

    # FIXED: Source mask - True for PAD tokens
    src_mask = (src == eng_stoi["<pad>"])

    # Initialize beam
    beams = [([ger_stoi["<bos>"]], 0.0)]

    with torch.no_grad():
        for step in range(max_len - 1):
            all_candidates = []

            for seq, score in beams:
                if seq[-1] == ger_stoi["<eos>"]:
                    all_candidates.append((seq, score))
                    continue

                seq_len = len(seq)

                # Pad target to max_len
                tgt_padded = seq + [ger_stoi["<pad>"]] * (max_len - seq_len)
                tgt = torch.tensor([tgt_padded]).to(device)

                # FIXED: Target mask - True for PAD tokens
                tgt_mask = (tgt == ger_stoi["<pad>"])

                with torch.cuda.amp.autocast():
                    output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

                # Get logits for last valid position
                logits = output[0, seq_len - 1]
                log_probs = torch.log_softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(log_probs, beam_width)

                for prob, idx in zip(top_probs, top_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + prob.item()
                    all_candidates.append((new_seq, new_score))

            # Keep top beam_width sequences
            beams = sorted(all_candidates, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]

            if all(seq[-1] == ger_stoi["<eos>"] for seq, _ in beams):
                break

    # Get best sequence
    best_seq = beams[0][0]
    ger_itos = ger_tokens.get_itos()

    # Convert to words
    translated_tokens = []
    for idx in best_seq[1:]:
        if idx == ger_stoi["<eos>"]:
            break
        token = ger_itos[idx]
        if token not in ["<bos>", "<eos>", "<pad>", "<unk>"]:
            translated_tokens.append(token)

    return ' '.join(translated_tokens) if translated_tokens else "<empty>"


# Test
test_sentences = [
    "a dog is running in the park",
    "the cat is sleeping on the bed",
    "two people are walking together",
    "children are playing with a ball"
]

print("=" * 60)
print("GREEDY DECODING")
print("=" * 60)

for sentence in test_sentences:
    translation = translate_sentence_greedy(sentence, model, eng_tokens, ger_tokens, device)
    print(f"\nEnglish:  {sentence}")
    print(f"German:   {translation}")

print("\n" + "=" * 60)
print("BEAM SEARCH (width=5)")
print("=" * 60)

for sentence in test_sentences:
    translation = translate_sentence_beam(sentence, model, eng_tokens, ger_tokens, device, beam_width=5)
    print(f"\nEnglish:  {sentence}")
    print(f"German:   {translation}")