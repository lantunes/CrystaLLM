import sys
sys.path.append(".")
import os
import time
import gzip
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import GPTConfig, GPT
from model_regression import GPTRegressor

from sklearn.metrics import r2_score, mean_absolute_error
from lib import abs_r_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
predictions_fname = ""
pretrained_model_dir = ""
pretraining_data_dir = ""
regression_dataset = ""
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
eval_interval = 250
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# system
device = "cuda" # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16" # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
init_from_pretrained = True  # whether to init regression model weights from pretrained model
freeze_transformer_weights = True  # whether to freeze the Transformer weights during fine-tuning
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(THIS_DIR, 'configurator.py')).read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
seed_offset = 0

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu" # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load the dataset
with gzip.open(regression_dataset, "rb") as f:
    data = pickle.load(f)
# convert the data to PyTorch tensors
for key in data:
    data[key] = torch.tensor(data[key], device=device_type)
# create PyTorch DataLoaders
train_dataset = TensorDataset(data["X_train"], data["y_train"])
val_dataset = TensorDataset(data["X_val"], data["y_val"])
test_dataset = TensorDataset(data["X_test"], data["y_test"])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(pretraining_data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# init regression model that will be fine-tuned
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
model_args["vocab_size"] = meta_vocab_size
gptconf = GPTConfig(**model_args)
regression_model = GPTRegressor(gptconf)

if init_from_pretrained:
    # load the pre-trained autoregressive model
    ckpt_path = os.path.join(pretrained_model_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    pretrained_model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    pretrained_model.load_state_dict(state_dict)

    # initialize the regression model from the pretrained model
    regression_model.init_from_pretrained(pretrained_model)

if freeze_transformer_weights:
    regression_model.freeze_transformer_weights()

# crop down the model block size if desired, using model surgery
if block_size < regression_model.config.block_size:
    regression_model.crop_block_size(block_size)
    model_args["block_size"] = block_size # so that the checkpoint will have the right value
regression_model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = regression_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = regression_model
    model = torch.compile(regression_model) # requires PyTorch 2.0

iter_num = 0
best_val_loss = float('inf')

# training loop
while True:
    start_time = time.time()

    # Train the model
    regression_model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            outputs, loss = regression_model(inputs, targets)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Print the training and time taken for the iteration
    elapsed_time = time.time() - start_time
    print(f"iter {iter_num}: time {elapsed_time:.2f}s - train loss: {train_loss:.4f}")

    if iter_num % eval_interval == 0:
        # Evaluate the model on the validation set
        regression_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                with ctx:
                    outputs, loss = regression_model(inputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print the validation loss and time taken for the iteration
        elapsed_time = time.time() - start_time
        print(f"step {iter_num}: time {elapsed_time:.2f}s - val loss: {val_loss:.4f}")

        # Save checkpoint if there is an improvement in validation loss
        if val_loss < best_val_loss:
            checkpoint = {
                'model': regression_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            best_val_loss = val_loss

    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# Evaluate the model on the test set

# load the best checkpoint
checkpoint = torch.load(os.path.join(out_dir, "ckpt.pt"))
regression_model.load_state_dict(checkpoint["model"])

# evaluate the model on the test set
regression_model.eval()
test_loss = 0.0
predictions = []
true_values = []
with torch.no_grad():
    for inputs, targets in test_loader:
        with ctx:
            outputs, loss = regression_model(inputs, targets)
        test_loss += loss.item()
        predictions.extend(outputs.cpu().numpy())
        true_values.extend(targets.cpu().numpy())

test_loss /= len(test_loader)

print(f"Test Loss: {test_loss:.4f}")

predictions = np.array(predictions)
true_values = np.array(true_values)
if predictions.ndim > 1:
    predictions = predictions.flatten()
if true_values.ndim > 1:
    true_values = true_values.flatten()

r2 = r2_score(true_values, predictions)
mae = mean_absolute_error(true_values, predictions)
abs_r = abs_r_score(true_values, predictions)

print(f"R2: {r2:.4f}, MAE: {mae:.4f}, |R|: {abs_r:.4f}")

print(f"saving test set predictions to {predictions_fname} ...")
combined_data = np.column_stack((predictions, true_values))
np.savetxt(predictions_fname, combined_data, delimiter=",")
