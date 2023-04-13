out_dir = "out/vpfu_model_1"
predictions_fname = "out/vpfu_model_1.predictions.csv"

regression_dataset = "out/mp_oqmd_cifs_semisymm_Z_props__vpfu.pkl.gz"
pretrained_model_dir = "out/cif_model_19"
pretraining_data_dir = "out/mp_oqmd_cifs_semisymm_Z_props"
batch_size = 32
block_size = 1024  # context of up to `block_size` previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 200
eval_interval = 20
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

init_from_pretrained = True
freeze_transformer_weights = True
