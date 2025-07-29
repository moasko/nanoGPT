import os
import pickle
import numpy as np

# Paramètres
input_file_path = 'data/dataset/input.txt'
output_dir = 'data/dataset'
train_fraction = 0.9

# Lire tout le texte
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Créer vocabulaire unique
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"{vocab_size=}, {len(data)=}")

# Mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Sauvegarder le vocab
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# Découpe train/val
n = len(data)
train_data = data[:int(n * train_fraction)]
val_data = data[int(n * train_fraction):]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

print("✅ Données préparées :", output_dir)
