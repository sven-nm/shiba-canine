from shiba.model import Shiba
from shiba.codepoint_tokenizer import CodepointTokenizer

import torch

a = torch.load('/scratch/sven/canine/output/checkpoint-75000/model.safetensors')