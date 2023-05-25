import os
import json
import torch
from ts.torch_handler.base_handler import BaseHandler
from contextlib import nullcontext
import logging

from model import GPTConfig, GPT

from lib import get_cif_tokenizer

logger = logging.getLogger(__name__)


class NanoGPTHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        self.model = None

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        serve_config_path = os.path.join(model_dir, "serve_config.json")
        if os.path.isfile(serve_config_path):
            with open(serve_config_path) as serve_config_file:
                serve_config = json.load(serve_config_file)
            logger.info(f"serve config: {serve_config}")
        else:
            logger.warning("Missing the serve_config.json file.")

        self.device = serve_config["device"]
        self.num_samples = serve_config["num_samples"]
        self.max_new_tokens = serve_config["max_new_tokens"]
        self.temperature = serve_config["temperature"]
        self.top_k = serve_config["top_k"]
        self.symmetrized = serve_config["symmetrized"]
        self.includes_props = serve_config["includes_props"]
        self.compile = serve_config["compile"]

        self.tokenizer = get_cif_tokenizer(symmetrized=self.symmetrized, includes_props=self.includes_props)
        self.encode = self.tokenizer.encode
        self.decode = self.tokenizer.decode

        self.ctx = nullcontext() if self.device == "cpu" else torch.amp.autocast(device_type=self.device,
                                                                                 dtype=torch.float32)

        checkpoint = torch.load("ckpt.pt", map_location=self.device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        self.model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        if self.compile:
            self.model = torch.compile(self.model)

    def preprocess(self, data):
        text = data[0].get("data") or data[0].get("body")
        decoded_text = text.decode("utf-8").replace('"', "")
        # NOTE: decoded_text should be a non-reduced formula, like "Na1Cl1"
        logger.info(f"decoded text received: {decoded_text}")
        return f"data_{decoded_text}\n"

    def inference(self, data, *args, **kwargs):
        start_ids = self.encode(self.tokenizer.tokenize_cif(data))
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])

        inference_output = []
        with torch.no_grad():
            with self.ctx:
                for k in range(self.num_samples):
                    y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k,
                                            symmetrized=self.symmetrized, includes_props=self.includes_props)
                    generated_content = self.decode(y[0].tolist())
                    inference_output.append({
                        "input": data,
                        "generated": generated_content,
                    })

        return [json.dumps(inference_output)]

    def postprocess(self, data):
        return data
