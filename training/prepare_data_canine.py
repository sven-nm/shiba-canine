from pathlib import Path
import re
import typing
import spacy

import jsonlines
from transformers import CanineTokenizer



def accumulate_sentences(lines: typing.Iterable[str], max_length: int, output_path) -> None:
    # Start by opening the output file
    outfile = jsonlines.open(output_path, 'w')

    accumulated_sentences = []
    current_sentence = ''
    for line in lines:
        # Remove all invisible characters
        line = re.sub(r'\s+', ' ', line)
        line = line.strip()

        if len(line) == 0:
            continue

        if len(line) > max_length:
            for sentence in sentence_splitter(line).sents:
                sentence = sentence.text.strip()
                if len(current_sentence) + len(sentence) > max_length:
                    tokenized = tokenizer(current_sentence, add_special_tokens=False, padding=True, truncation=True, pad_to_multiple_of=2048)
                    outfile.write(tokenized.data)
                    current_sentence = sentence
                    if len(current_sentence) > max_length:
                        tokenized = tokenizer(current_sentence, add_special_tokens=False, padding=True, truncation=True)
                        outfile.write(tokenized.data)
                        current_sentence = ''
                else:
                    current_sentence += sentence + ' '

        if len(current_sentence) + len(line) > max_length:
            tokenized = tokenizer(current_sentence, add_special_tokens=False, padding=True, truncation=True, pad_to_multiple_of=2048)
            outfile.write(tokenized.data)
            current_sentence = line
            if len(current_sentence) > max_length:
                tokenized = tokenizer(current_sentence, add_special_tokens=False, padding=True, truncation=True)
                outfile.write(tokenized.data)
                current_sentence = ''
        else:
            current_sentence += line + ' '

    tokenized = tokenizer(current_sentence, add_special_tokens=False, padding=True, truncation=True, pad_to_multiple_of=2048)
    outfile.write(tokenized.data)
    outfile.close()



if __name__ == '__main__':
    root_data_dir = Path('/mnt/ajmcdata1/data')
    output_dir = Path('/scratch/sven/canine/pre_training_data')
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_splitter = spacy.load("xx_sent_ud_sm")  # Load multilingual model
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    for txt_file in root_data_dir.rglob('*/plaintext.txt'):
        print('Handling ', txt_file)
        with open(txt_file, 'r', encoding='utf-8') as f:
            accumulate_sentences(f, 2048, output_dir / f'{txt_file.parent.name}.jsonl')
        break
