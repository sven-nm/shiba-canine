from pathlib import Path
import re
import spacy

import jsonlines
from transformers import CanineTokenizer
import unicodedata
from tqdm import tqdm

max_length = 2048

if __name__ == '__main__':

    # TODO: MISSING: ia_commentaries
    cleaned_corpora = {'agoraclass': 1,
                       'Brill-KIEM-data': 1,
                       'corpus_scriptorum_latinorum': 1,
                       'corpus_thomisticum': 1,
                       'forum_romanum': 1,
                       'JSTOR-dataset-2021': 1,
                       'mediterranee_antique': 1,
                       'remacle': 1,
                       'the_latin_library': 1,
                       'wiki_el': 1,
                       'wiki_en': 1,
                       'wiki_fr': 1,
                       'wiki_it': 3,
                       'wiki_la': 1,
                       'canonical-greekLit': 1,
                       'canonical-latinLit': 1,
                       'EpibauCorpus': 2,
                       'First1KGreek': 1,
                       'logeion_greek': 4,
                       'logeion_latin': 1,
                       'persee': 1,
                       'perseus_legacy': 1,
                       'perseus_secondary': 4,
                       'riemenschneider_born_digital': 1,
                       'riemenschneider_internet_archive': 1,
                       'propylaeum_BOOKS': 3,
                       'propylaeum_DOK': 3,
                       }

    txt_paths = {corpus: Path(f'/mnt/ajmcdata1/data/{corpus}/cleantext.txt') for corpus in cleaned_corpora}

    root_data_dir = Path('/mnt/ajmcdata1/data/')
    output_dir = Path('/scratch/sven/canine/pre_training_data')
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_splitter = spacy.load("xx_sent_ud_sm")  # Load multilingual model
    sentence_splitter.max_length = 6046153  # Increase max length to avoid errors
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    for corpus_id, txt_path in txt_paths.items():
        print('Handling ', txt_path)

        output_path = output_dir / f'{corpus_id}.jsonl'
        if output_path.exists():
            print(f'{output_path} already exists, skipping')
            continue

        outfile = jsonlines.open(output_path, 'w')

        current_example = ''
        excluded_long_sentences = 0
        for line in tqdm(txt_path.open('r', encoding='utf-8')):
            # Remove all invisible characters
            line = unicodedata.normalize("NFC", line)
            line = re.sub(r'\s+', ' ', line)
            line = line.strip()

            if len(line) < 50:
                continue

            # If adding the line to the current sentence does not exceed the max length, simply add it
            if len(current_example) + len(line) < max_length:
                current_example += line + ' '
                continue

            else:  # current_example + line > max_length
                # Split the current sentence into sentences
                for sentence in sentence_splitter(line).sents:
                    sentence = sentence.text.strip()

                    if len(sentence) > max_length:
                        sentence = sentence[:max_length]

                    if len(current_example) + len(sentence) < max_length:
                        current_example += sentence + ' '
                    else:
                        tokenized = tokenizer(current_example, add_special_tokens=False, truncation=True)
                        outfile.write({'input_ids': tokenized['input_ids']})
                        current_example = sentence + ' '

        #######################################################
        print(f'Excluded {excluded_long_sentences} sentences that were too long')
        tokenized = tokenizer(current_example, add_special_tokens=False, truncation=True)
        outfile.write({'input_ids': tokenized['input_ids']})
        outfile.close()

        # Copy the files to weight the corpora by size
        if cleaned_corpora[corpus_id] > 1:
            for i in range(1, cleaned_corpora[corpus_id]):
                (output_dir / f'{corpus_id}_{i}.jsonl').write_text(output_path.read_text(encoding='utf-8'), encoding='utf-8')