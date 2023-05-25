import os
import shutil
import torch

from nanoGPT.model import GPTConfig, GPT

"""
TODO running this script as of 2023-05-24 results in:
"Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 14 is not supported."
See this issue: https://github.com/pytorch/pytorch/issues/97262
Apparently this is fixed in torch nightly as of 2023-05-24.
"""
if __name__ == '__main__':

    out_dir = "../out/cif_model_19"
    save_dir = "../out/cif_model_19_onnx"
    device = "cpu"

    # make the output directory if it doesn't exist
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # load the model
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()

    # a dummy input that matches the expected input size
    dummy_input = torch.randint(0, 100, (1, 5))

    # export the model to an ONNX file
    torch.onnx.export(model, dummy_input, f"{save_dir}/model.onnx")
