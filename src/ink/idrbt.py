"""
IDRBT Cheque Image Dataset helpers.

The dataset has 112 cheques (300 dpi TIFF) and DATA.xlsx describing each
one: writer + pen of the original filling (pen 1) and writer + pen of the
later alteration (pen 2), plus the exact text each pen wrote, e.g.

    pen 1: "Satish kumar, twenty five thousand, 25000"
    pen 2: "apparao, one lakh, 1"
    -> cheque reads "Apparao Satish Kumar / one lakh Twenty Five Thousand / 1,25000/-"

Pen 2 always *prepends* to the amount (words and figures), so the leading
k digits / words of the amount were written by pen 2.  This gives real,
pen-labelled ink samples without any manual annotation.
"""
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

_BANK_KEYS = {'AXIS BANK': 'axis', 'ICICI BANK': 'icici',
              'SYNDICATE BANK': 'syndicate', 'CANARA BANK': 'canara'}

# Pens 1-7 are blue, 8-14 black (sheet "PEN DETAILS")
PEN_COLOURS = {i: ('blue' if i <= 7 else 'black') for i in range(1, 15)}

_NUMBER_WORDS = [
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fourty',
    'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand',
    'lakhs', 'lakh', 'lacks', 'lack', 'lacs', 'lac', 'crores', 'crore', '&', 'and',
]


def find_dataset_dir(raw_dir=None):
    """Locate the folder containing DATA.xlsx (searches data/raw recursively)."""
    raw_dir = raw_dir or config.RAW_DATA_DIR
    hits = glob.glob(os.path.join(raw_dir, '**', 'DATA.xlsx'), recursive=True)
    if not hits:
        raise FileNotFoundError(
            f"IDRBT dataset not found under {raw_dir}. Extract "
            "IDRBT_Cheque_Image_Dataset.zip into data/raw/.")
    return os.path.dirname(hits[0])


def count_amount_words(text):
    """Number of number-words in an amount phrase ('onecrore' -> 2)."""
    s = str(text).lower()
    count = 0
    for w in _NUMBER_WORDS:
        n = len(re.findall(re.escape(w), s))
        if n:
            count += n
            s = s.replace(w, ' ')
    return max(count, 1)


def _split3(text):
    parts = [p.strip() for p in str(text).split(',')]
    while len(parts) < 3:
        parts.append('')
    return parts[0], ', '.join(parts[1:-1]), parts[-1]


def load_metadata(raw_dir=None):
    """
    Returns a DataFrame with one row per cheque:
        cheque, path, bank, bank_key, pen1, pen2, user1, user2,
        text1, text2, payee1, payee2, words1, words2, digits1, digits2,
        k_digits (pen-2 leading digits), k_words (pen-2 leading words)
    """
    ds = find_dataset_dir(raw_dir)
    df = pd.read_excel(os.path.join(ds, 'DATA.xlsx'), sheet_name=0)
    df.columns = ['cheque', 'user1', 'pen1', 'text1', 'user2', 'pen2', 'text2', 'bank']
    df = df.dropna(subset=['cheque'])
    df['cheque'] = df.cheque.astype(int)
    df['bank'] = df.bank.str.strip()
    df['bank_key'] = df.bank.map(_BANK_KEYS)

    files = {}
    for f in glob.glob(os.path.join(ds, '**', '*.tif'), recursive=True):
        m = re.search(r'(\d+)\.tif$', f, re.I)
        if m:
            files[int(m.group(1))] = f
    df['path'] = df.cheque.map(files)
    df = df.dropna(subset=['path']).reset_index(drop=True)

    for i in (1, 2):
        sp = df[f'text{i}'].map(_split3)
        df[f'payee{i}'] = sp.map(lambda t: t[0])
        df[f'words{i}'] = sp.map(lambda t: t[1])
        df[f'digits{i}'] = sp.map(lambda t: re.sub(r'\D', '', t[2]))
    df['k_digits'] = df.digits2.str.len().clip(lower=1)
    df['k_words'] = df.words2.map(count_amount_words)
    df['colour'] = df.pen1.map(PEN_COLOURS)
    return df
