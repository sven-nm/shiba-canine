import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["WANDB_DISABLED"] = "true"

from transformers import CanineModel, CanineTokenizer

from typing import Optional, Tuple, Dict
import torch

import transformers
from transformers import HfArgumentParser, Trainer
from datasets import load_dataset, Dataset

import typing
from training.helpers import DataArguments, ShibaTrainingArguments, get_model_hyperparams
from training.masking import RandomSpanMaskingDataCollator, RandomMaskingDataCollator, random_mask
from shiba import ShibaForAutoregressiveLanguageModeling, CodepointTokenizer


MAX_GR_CODEPOINT = 8191

class CanineForAutoregressiveLanguageModeling(ShibaForAutoregressiveLanguageModeling):

    def __init__(self, vocab_size: int = 16533, **kwargs):
        super().__init__(vocab_size=vocab_size, **kwargs)  # todo fix this
        self.shiba_model = CanineModel.from_pretrained("google/canine-c")


    # We need to override the forward method to use the Canine model, as canine also uses token_type_ids
    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor],
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                predict_indices: torch.Tensor) -> Tuple:

        # Here we want canine's last hidden state BEFORE the final transformer
        upsampled_embeddings = self.shiba_model(input_ids,
                                                attention_mask,
                                                token_type_ids,
                                                output_hidden_states=True).hidden_states[-2]

        # this is MLM of some kind - we don't need to do the final CLS computation and we can only do the
        # final transformer for the positions we're predicting
        embeddings_for_pred = torch.stack([upsampled_embeddings[i, predict_indices[i], :]
                                           for i in range(upsampled_embeddings.shape[0])])

        # no attention mask because we are presumably not trying to predict padding
        final_embeddings = self.shiba_model.final_char_encoder(embeddings_for_pred).last_hidden_state

        causal_mask = self._get_causal_mask(final_embeddings)
        autoregressive_char_seq = self.autregressive_encoder(final_embeddings,
                                                             src_mask=causal_mask)

        lm_hidden_states = self.lm_layer(autoregressive_char_seq)
        char_probs = self.log_softmax(lm_hidden_states)

        return self._compute_loss(final_embeddings, char_probs, predict_indices, labels)


class CanineRandomMaskingDataCollator:

    def __init__(self, replacement_range: range):
        self.replacement_range = replacement_range

    def __call__(self, batch: typing.List[dict]) -> Dict[str, torch.Tensor]:
        # 🚧 Here we change the tokenizer.pad as it is not available in CanineTokenizer
        batch = {'input_ids': torch.tensor([x['input_ids'] + [0] * (2048 - len(x['input_ids'])) for x in batch]),
                 'attention_mask': torch.tensor([[1] * len(x['input_ids']) + [0] * (2048 - len(x['input_ids'])) for x in batch]),
                 'token_type_ids': torch.zeros((len(batch), 2048)).long(),
                 }

        input_ids, labels, masked_indices = random_mask(batch['input_ids'],
                                                        batch['attention_mask'],
                                                        min_char_replacement=self.replacement_range.start,
                                                        max_char_replacement=self.replacement_range.stop)

        batch.update({  # Token type ids are going to be kept as 0
            'input_ids': input_ids,
            'labels': labels,
            'predict_indices': masked_indices
        })

        return batch


# Note: This is already assuming that the data is tokenized and stored as {"input_ids": [int], "attention_mask": [int]} jsonl files
def prepare_data(args: DataArguments) -> Tuple[Dataset, Dataset]:
    all_data = load_dataset('json', data_dir=args.data_dir, split='train')
    data_dict = all_data.train_test_split(train_size=0.98, seed=42)
    training_data = data_dict['train']
    training_data = training_data.shuffle(seed=42)
    dev_data = data_dict['test']
    dev_data = dev_data.shuffle(seed=42)
    return training_data, dev_data


def main():
    transformers.logging.set_verbosity_info()
    parser = HfArgumentParser((DataArguments, ShibaTrainingArguments))

    data_args, training_args = parser.parse_dict({'data_dir': '/home/najem/dhlab-data/data/najem-data/canine/pre_training_data',
                                                  'output_dir': '/home/najem/dhlab-data/data/najem-data/canine/output',
                                                  'masking_type': 'rand_char',
                                                  })

    # torch.cuda.set_device(1)
    # Set our own training parameters
    training_args.num_train_epochs = 10

    # 🚧 Here we change the tokenizer to CanineTokenizer
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    if training_args.masking_type == 'bpe_span':
        print('BPE based-span masking')
        data_collator = RandomSpanMaskingDataCollator(tokenizer, True)
    elif training_args.masking_type == 'rand_span':
        print('Random span masking')
        data_collator = RandomSpanMaskingDataCollator(tokenizer, False)

    # 🚧 Here we change the data_collator to CanineRandomMaskingDataCollator
    elif training_args.masking_type == 'rand_char':
        print('Random character masking')
        # char range: https://stackoverflow.com/a/30200250/4243650
        # we aren't including half width stuff
        data_collator = CanineRandomMaskingDataCollator(range(1, MAX_GR_CODEPOINT))
    else:
        raise RuntimeError('Unknown masking type')

    training_args.logging_dir = training_args.output_dir
    training_data, dev_data = prepare_data(data_args)
    model_hyperparams = get_model_hyperparams(training_args)

    model = CanineForAutoregressiveLanguageModeling(MAX_GR_CODEPOINT, **model_hyperparams)

    checkpoint_dir = None
    if training_args.resume_from_checkpoint:
        if training_args.load_only_model:
            model.load_state_dict(torch.load(training_args.resume_from_checkpoint))
        else:
            checkpoint_dir = training_args.resume_from_checkpoint
    os.environ['WANDB_PROJECT'] = 'shiba'

    print(training_args)
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=training_data,
                      eval_dataset=dev_data,
                      )

    trainer.train(resume_from_checkpoint=checkpoint_dir)


if __name__ == '__main__':
    main()
