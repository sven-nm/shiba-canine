"""The goal of this script is to prepare data for any model for pre-training, in oder to make it run with HF run_mlm.py example."""

from pathlib import Path
import re
import spacy
from transformers import AutoTokenizer
import unicodedata
from tqdm import tqdm

max_length = 512
model_name = 'FacebookAI/xlm-roberta-base'
output_dir = Path(f'/scratch/sven/{model_name.split("/")[-1]}/pre_training_data')
output_dir.mkdir(parents=True, exist_ok=True)


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
                       'canonical-greekLit': 2,
                       'canonical-latinLit': 1,
                       'EpibauCorpus': 4,
                       'First1KGreek': 1,
                       'logeion_greek': 5,
                       'logeion_latin': 1,
                       'persee': 1,
                       'perseus_legacy': 1,
                       'perseus_secondary': 8,
                       'riemenschneider_born_digital': 1,
                       'riemenschneider_internet_archive': 1,
                       'propylaeum_BOOKS': 4,
                       'propylaeum_DOK': 8,
                       }

    txt_paths = {corpus: Path(f'/mnt/ajmcdata1/data/{corpus}/cleantext.txt') for corpus in cleaned_corpora}


    sentence_splitter = spacy.load("xx_sent_ud_sm")  # Load multilingual model
    sentence_splitter.max_length = 6046153  # Increase max length to avoid errors
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    output_path = output_dir / 'train.txt'
    outfile = output_path.open('w', encoding='utf-8')

    for corpus_id, txt_path in txt_paths.items():
        print('Handling ', txt_path)

        current_example = ''
        current_example_length = 0

        for line in tqdm(txt_path.open('r', encoding='utf-8')):
            # Remove all invisible characters
            line = unicodedata.normalize("NFC", line)
            line = re.sub(r'\s+', ' ', line)
            line = line.strip()

            if len(line) < 50:
                continue

            for sentence in sentence_splitter(line).sents:
                sentence = sentence.text.strip()
                tokenized = tokenizer(sentence, padding=False, truncation=True, add_special_tokens=False)

                if current_example_length + len(tokenized['input_ids']) <= max_length-2:  # We need to add 2 for the special tokens
                    current_example += sentence + ' '
                    current_example_length += len(tokenized['input_ids'])
                else:
                    for _ in range(cleaned_corpora[corpus_id]):
                        outfile.write(current_example+'\n')
                    current_example = sentence + ' '
                    current_example_length = len(tokenized['input_ids'])

        #######################################################
        outfile.write(current_example)

    outfile.close()