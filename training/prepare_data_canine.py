from pathlib import Path
import re
import spacy

import jsonlines
from transformers import CanineTokenizer
import unicodedata
from tqdm import tqdm

max_length = 2048

if __name__ == '__main__':

    # TODO: MISSING: ia_commentaries, propylaeum_BOOKS, propylaeum_DOK
    # TODO: Weight the corpora by size ?
    cleaned_corpora = ['agoraclass', 'Brill-KIEM-data', 'corpus_scriptorum_latinorum', 'corpus_thomisticum', 'forum_romanum', 'JSTOR-dataset-2021',
                       'mediterranee_antique', 'remacle', 'the_latin_library', 'wiki_el', 'wiki_en', 'wiki_fr', 'wiki_it', 'wiki_la',
                       'canonical-greekLit', 'canonical-latinLit', 'EpibauCorpus', 'First1KGreek', 'logeion_greek', 'logeion_latin', 'persee',
                       'perseus_legacy', 'perseus_secondary', 'riemenschneider_born_digital', 'riemenschneider_internet_archive'
                       ]

    txt_paths = {corpus: Path(f'/mnt/ajmcdata1/data/{corpus}/cleantext.txt') for corpus in cleaned_corpora}

    root_data_dir = Path('/mnt/ajmcdata1/data/')
    output_dir = Path('/scratch/sven/canine/pre_training_data')
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_splitter = spacy.load("xx_sent_ud_sm")  # Load multilingual model
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    for corpus_id, txt_path in txt_paths.items():
        print('Handling ', txt_path)

        output_path = output_dir / f'{corpus_id}.jsonl'
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
                        excluded_long_sentences += 1

                    else:  # len(sentence) < max_length
                        if len(current_example) + len(sentence) < max_length:
                            current_example += sentence + ' '
                        else:
                            tokenized = tokenizer(current_example, add_special_tokens=False, truncation=True)
                            outfile.write({'input_ids': tokenized['input_ids']})
                            # print('-----------------------------', len(current_example), '-----------------------------')
                            # print(current_example)
                            current_example = sentence + ' '

        #######################################################
        print(f'Excluded {excluded_long_sentences} sentences that were too long')
        tokenized = tokenizer(current_example, add_special_tokens=False, truncation=True)
        outfile.write({'input_ids': tokenized['input_ids']})
        outfile.close()
