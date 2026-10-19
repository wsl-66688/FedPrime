import json
import pyarrow as pa
import random
import os
import re
import tempfile

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile(r"(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]

def normalize_word(token):
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) != None
        ):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token

def merge_keys(d, old_key, existing_key):
    if old_key in d and existing_key in d:
        d[existing_key].extend(d.pop(old_key))

def make_arrow(root, dataset_root, single_plot=False, missing_type=None, batch_size=256):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    image_root = os.path.join(root, 'images')
    
    with open(f"{root}/class_idx.json", "r") as fp:
        FOOD_CLASS_DICT = json.load(fp)
        
    with open(f"{root}/text.json", "r") as fp:
        text_dir = json.load(fp)
        
    with open(f"{root}/split.json", "r") as fp:
        split_sets = json.load(fp)
    
    merge_keys(split_sets, 'val', 'train')

    schema = pa.schema([
        ('image', pa.binary()),
        ('text', pa.list_(pa.string())),
        ('label', pa.int64()),
        ('image_id', pa.string()),
        ('split', pa.string()),
    ])
    os.makedirs(dataset_root, exist_ok=True)

    def write_batch(writer, rows):
        columns = {
            name: [row[index] for row in rows]
            for index, name in enumerate(schema.names)
        }
        writer.write_table(pa.Table.from_pydict(columns, schema=schema))

    for split, samples in split_sets.items():
        split_type = 'train' if split != 'test' else 'test'
        data_list = []
        written_rows = 0
        skipped_rows = 0
        output_path = os.path.join(dataset_root, f'food101_{split}.arrow')
        # Keep the existing output intact until the complete replacement is ready.
        with tempfile.NamedTemporaryFile(
            prefix=f'.food101_{split}.', suffix='.arrow.tmp',
            dir=dataset_root, delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
        try:
            with pa.OSFile(temporary_path, 'wb') as sink:
                with pa.RecordBatchFileWriter(sink, schema) as writer:
                    for sample in tqdm(samples):
                        if sample not in text_dir:
                            print("ignore no text data: ", sample)
                            skipped_rows += 1
                            continue
                        cls = sample[:sample.rindex('_')]
                        label = FOOD_CLASS_DICT[cls]
                        image_path = os.path.join(image_root, split_type, cls, sample)
                        with open(image_path, 'rb') as fp:
                            binary = fp.read()
                        data_list.append((binary, [text_dir[sample]], label, sample, split))
                        if len(data_list) >= batch_size:
                            write_batch(writer, data_list)
                            written_rows += len(data_list)
                            data_list.clear()
                    if data_list:
                        write_batch(writer, data_list)
                        written_rows += len(data_list)
                        data_list.clear()

            # Check the footer, schema and row count before publishing the output.
            with pa.OSFile(temporary_path, 'rb') as source:
                reader = pa.ipc.RecordBatchFileReader(source)
                actual_rows = sum(
                    reader.get_batch(index).num_rows
                    for index in range(reader.num_record_batches)
                )
                if not reader.schema.equals(schema) or actual_rows != written_rows:
                    raise ValueError(f'Incomplete Arrow output for {split}')
                del reader
            os.replace(temporary_path, output_path)
            print(f'Saved {output_path}: {written_rows} rows, skipped {skipped_rows}')
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)


if __name__ == '__main__':
    root = 'benchmark/RAW_DATA/FOOD101'
    dataset_saved = 'benchmark/RAW_DATA/FOOD101/generate_arrows'
    make_arrow(root, dataset_saved)
