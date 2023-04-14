out_dir = "out/vpfu_model_4"
predictions_fname = "out/vpfu_model_4.predictions.csv"

regression_dataset = "out/mp_oqmd_cifs_semisymm_Z_props__vpfu.pkl.gz"
pretrained_model_dir = "out/cif_model_19"
pretraining_data_dir = "out/mp_oqmd_cifs_semisymm_Z_props"
batch_size = 32
block_size = 1024  # context of up to `block_size` previous characters

n_layer = 8
n_head = 8
n_embd = 512
additional_layer_size = 1024
dropout = 0.1

learning_rate = 1e-5
max_iters = 100
eval_interval = 10
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

init_from_pretrained = True
freeze_transformer_weights = True
