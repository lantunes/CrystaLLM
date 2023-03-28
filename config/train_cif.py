out_dir = 'out/cif_model'
eval_interval = 250  # how often to evaluate against the validation set
eval_iters = 200
log_interval = 1  # how often to print to the console (1 = every iteration)

# whether to always save a checkpoint
always_save_checkpoint = False

wandb_log = False
wandb_project = 'cif'
wandb_run_name = 'mini-gpt'

dataset = '../out/mp_cifs'
batch_size = 64
block_size = 2048  # context of up to `block_size` previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
