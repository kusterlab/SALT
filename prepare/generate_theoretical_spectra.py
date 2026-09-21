"""Generate the target theoretical spectra library for RNA ladder fragments.

Enumerates every 2-5 nt sequence over {A, U, G, C, S} containing exactly one crosslink
site (X), then for each sequence derives its XL / secondary / diagnostic fragment sets
(by cutting the phosphodiester-linked sequence at "p" positions), converts fragments to
composition labels, and looks up each label's m/z in the ladder mass reference table.

Input (read from this directory):
  massdiff_adduct_annot_5nt_RNAladder_diagnpeaks.csv

Output (written to this directory):
  theoretical_spectra_{MAX_LENGTH}nt.csv   (MAX_LENGTH = 5 by default)

Run manually (not part of the pipeline runner) whenever the ladder chemistry or mass
reference table changes; other prepare/ scripts (e.g. generate_decoy_theoretical_spectra.py)
import this module to reuse its sequence/fragment/label logic.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import warnings

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from salt.utils import arrays_to_json, json_to_arrays

ref_mass_table = pd.read_csv(_HERE / "massdiff_adduct_annot_5nt_RNAladder_diagnpeaks.csv")
ref_mass_table["round_1+"] = ref_mass_table["1+"].round(6) 

MAX_LENGTH = 5  



def generate_sequences_with_x(max_length: int = MAX_LENGTH) -> dict[int, list[str]]:
    """
    Generate all possible sequences with exactly one X and remaining positions filled with A/U/G/C/S.
    
    Parameters:
    max_length: maximum sequence length (generates lengths 2 to max_length inclusive)
    
    Returns:
    dict with length as key and list of sequences as value
    """
    bases = ['A', 'U', 'G', 'C', 'S']
    all_sequences = {}
    
    for length in range(2, max_length + 1):
        sequences = []
        
        # For each position where X can be placed
        for x_position in range(length):
            # Generate all combinations of bases for the remaining positions
            for base_combo in itertools.product(bases, repeat=length-1):
                # Insert X at the specified position
                seq_list = list(base_combo)
                seq_list.insert(x_position, 'X')
                sequences.append(''.join(seq_list))
        
        all_sequences[length] = sorted(sequences)
        print(f"Length {length}: {len(sequences)} sequences")
    
    return all_sequences


def get_p_fragments(seq_with_p: str) -> list[str]:
    """
    Get all possible p-fragments by cutting after each 'p'.
    Returns all substrings that span between cut points and touch either the start or end.
    Include the whole sequence itself.
    
    Example: 'pApApX' -> ['pAp', 'ApAp', 'pApApX', 'ApX', 'X', 'pApX']
    """
    # Find all 'p' positions
    p_positions = [i for i, char in enumerate(seq_with_p) if char == 'p']
    
    # Cut positions: after each 'p'
    cut_positions = [pos + 1 for pos in p_positions]
    
    # All boundaries (start, cuts, end)
    boundaries = sorted(set([0] + cut_positions + [len(seq_with_p)]))
    
    fragments = []
    for i in range(len(boundaries)):
        for j in range(i+1, len(boundaries)):
            start, end = boundaries[i], boundaries[j]
            fragment = seq_with_p[start:end]
            
            # Include if it touches start or end
            if start == 0 or end == len(seq_with_p):
                fragments.append(fragment)
    
    return fragments

def get_secondary_fragments(seq_with_p: str) -> list[str]:
    """
    Get all possible secondary fragments by cutting at every two-p combination.
    For each pair of p positions, make both cuts and generate the 3 resulting fragments.
    
    Example: 'pApUpGpCpX' with p at [0, 2, 4, 6, 8]
    Two-p at positions (0,2): cuts at 1,3 -> fragments: [0,1], [1,3], [3,10]
    """
    p_positions = [i for i, char in enumerate(seq_with_p) if char == 'p']
    
    if len(p_positions) < 2:
        return []
    
    fragments = set()
    seq_len = len(seq_with_p)
    
    # For each combination of two p's, make both cuts simultaneously
    for p1_idx, p2_idx in itertools.combinations(p_positions, 2):
        c1 = p1_idx + 1  # cut position after first p
        c2 = p2_idx + 1  # cut position after second p
        
        # Ensure c1 < c2
        if c1 > c2:
            c1, c2 = c2, c1
        
        # Generate the 3 fragments from simultaneous cuts at c1 and c2
        fragments.add(seq_with_p[0:c1])       # start to first cut
        fragments.add(seq_with_p[c1:c2])      # first cut to second cut
        fragments.add(seq_with_p[c2:seq_len]) # second cut to end
    
    return sorted(list(fragments))

def separate_fragments(fragments: Iterable[str]) -> tuple[list[str], list[str]]:
    """Separate fragments into X-containing and non-X-containing"""
    with_x = [f for f in fragments if 'X' in f]
    without_x = [f for f in fragments if 'X' not in f]
    return with_x, without_x

def convert_fragment_to_label(fragments: str | Iterable[str]) -> str | list[str]:
    """
    Convert fragment(s) to label format.
    Rules: If X present: X at start, rest of upper case letters sorted alphabetically
           If no X: just sort all letters alphabetically
    Special: +0P is omitted, +1P becomes +P
    
    Parameters:
    fragments: single fragment string or list of fragments
    
    Returns:
    single label string or list of label strings
    
    Example: 'pGpSpApX' -> 'XAGS+4P', 'pGpS' -> 'GS+2P', 'X' -> 'X'
    """
    def convert_one(fragment: str) -> str:
        p_count = fragment.count('p')
        letters = [c for c in fragment if c != 'p']
        
        if 'X' in letters:
            non_x_letters = sorted([c for c in letters if c != 'X'])
            label = 'X' + ''.join(non_x_letters)
        else:
            label = ''.join(sorted(letters))
        
        if p_count == 0:
            return label
        elif p_count == 1:
            return label + '+P'
        else:
            return label + f'+{p_count}P'
    
    if isinstance(fragments, str):
        return convert_one(fragments)
    else:
        return [convert_one(f) for f in fragments]


def lookup_labels_in_ref_table(
    labels: Iterable[str],
    ref_table: pd.DataFrame,
) -> tuple[list[str], list[float]]:
    """
    For each label in labels array, find matching rows in ref_table by simp_label,
    then return arrays of label and round_1+ values.
    Throws warning if any labels are not matched.
    """
    legible_labels = []
    masses = []
    missed_match = []

    for label in labels:
        matches = ref_table[ref_table['simp_label'] == label]
        if len(matches) > 0:
            legible_labels.extend(matches['label'].tolist())
            masses.extend(matches['round_1+'].tolist())
        else:
            missed_match.append(label)
    
    if missed_match:
        unique_missed = list(set(missed_match))
        warnings.warn(f"Labels not found in ref_table: {unique_missed}")
    
    return legible_labels, masses

sequences = generate_sequences_with_x()


total = sum(len(seqs) for seqs in sequences.values())
print(f"\nTotal sequences: {total}")

data = []
for length in sorted(sequences.keys()):
    for seq in sequences[length]:
        p_seq = 'p' + 'p'.join(seq)
        data.append({
            'sequence': seq, 
            'length': length,
            'p_sequence': p_seq,
            'all_fragments': get_p_fragments(p_seq),
            'secondary_fragments': get_secondary_fragments(p_seq)
        })

df = pd.DataFrame(data)

df[['XL_fragments', 'diagnostic_fragments']] = df['all_fragments'].apply(
    lambda frags: pd.Series(separate_fragments(frags))
)

df['diagnostic_fragments'] = df['diagnostic_fragments'].apply(
    lambda frags: [
        frag for frag in frags
        if frag != 'p' and sum(ch.isupper() for ch in frag) > 1
    ]
)

df['secondary_fragments'] = df.apply(
    lambda row: [
        frag for frag in row['secondary_fragments']
        if 'X' in frag and frag not in set(row['XL_fragments'])
    ],
    axis=1,
)


df['n_XL_fragments'] = df['XL_fragments'].apply(len)
df['n_secondary_fragments'] = df['secondary_fragments'].apply(len)
df['n_diagnostic_fragments'] = df['diagnostic_fragments'].apply(len)

# Convert fragments to labels
df['XL_simp_labels'] = df['XL_fragments'].apply(convert_fragment_to_label)
df['diagnostic_simp_labels'] = df['diagnostic_fragments'].apply(convert_fragment_to_label)
df['secondary_simp_labels'] = df['secondary_fragments'].apply(convert_fragment_to_label)


# Apply lookup to XL_simp_labels (simp_label -> label/round_1+)
df[['XL_labels', 'XL_fragment_masses']] = df['XL_simp_labels'].apply(
    lambda labels: pd.Series(lookup_labels_in_ref_table(labels, ref_mass_table))
)

# Apply lookup to diagnostic_simp_labels (simp_label -> label/round_1+)
df[['diagnostic_labels', 'diagnostic_fragment_masses']] = df['diagnostic_simp_labels'].apply(
    lambda labels: pd.Series(lookup_labels_in_ref_table(labels, ref_mass_table))
)

# Apply lookup to secondary_simp_labels (simp_label -> label/round_1+)
df[['secondary_labels', 'secondary_fragment_masses']] = df['secondary_simp_labels'].apply(
    lambda labels: pd.Series(lookup_labels_in_ref_table(labels, ref_mass_table))
)

# Enforce a stable, analysis-friendly column order.
column_order = [
    'sequence',
    'length',
    'p_sequence',
    'all_fragments',
    'XL_fragments',
    'secondary_fragments',
    'diagnostic_fragments',
    'n_XL_fragments',
    'n_secondary_fragments',
    'n_diagnostic_fragments',
    'XL_simp_labels',
    'secondary_simp_labels',
    'diagnostic_simp_labels',
    'XL_labels',
    'XL_fragment_masses',
    'secondary_labels',
    'secondary_fragment_masses',
    'diagnostic_labels',
    'diagnostic_fragment_masses',
]
df = df[column_order]



output_file = _HERE / f"theoretical_spectra_{MAX_LENGTH}nt.csv"
array_cols = [
    'all_fragments',
    'secondary_fragments',
    'XL_fragments',
    'diagnostic_fragments',
    'XL_simp_labels',
    'diagnostic_simp_labels',
    'secondary_simp_labels',
    'XL_labels',
    'diagnostic_labels',
    'secondary_labels',
    'XL_fragment_masses',
    'diagnostic_fragment_masses',
    'secondary_fragment_masses',
]

df_to_save = arrays_to_json(df, array_cols)
df_to_save.to_csv(output_file, index=False)

# Read back with JSON decoding so downstream scripts get list-like arrays.
df = pd.read_csv(output_file)
df = json_to_arrays(df, array_cols)
print(f"\nSaved to {output_file}")
