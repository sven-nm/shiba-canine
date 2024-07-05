from shiba.model import Shiba
from shiba.codepoint_tokenizer import CodepointTokenizer


tokenizer = CodepointTokenizer()



model = Shiba()

#%%

inputs =  ['coucou coucou coucou coucou coucou coucou coucou !', 'bla coucou']
inputs = tokenizer.encode_batch(inputs)
# inputs = tokenizer.pad(inputs)


outputs = model(**inputs)


# Shiba recieves an N_batch x 2048 tensor as input for both input_ids and attention_mask
# and returns an N_batch x 2048 x 768 tensor as output

#%%

import json
filepath = '/scratch/sven/canine/pre_training_data/testing_data.jsonl'

lengths = []
with open(filepath, 'r') as f:
    for line in f:
        data = json.loads(line)
        lengths.append(len(data['input_ids']))



#%%

text = ["HuggingFace is a company based in Paris and New York", "ηθγγινγ φ;ψε ισ ; ψο´π;γινε δηδι"]


