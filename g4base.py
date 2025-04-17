#!/usr/bin/env python3

#    Identification of potential RNA G-quadruplexes by G4RNA screener.
#    Copyright (C) 2018  Jean-Michel Garant
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import regex
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def cgcc_scorer(sequence):
    """
    Returns the cGcC score of a sequence.

    ****NOTE: The cGcC score does not consider flanking G/C like the G4Hunter****

    Returns a float.
    """
    G_score = 0
    C_score = 0
    for i in range(1, len(sequence) + 1):  # Replaced xrange with range
        G_score += len(regex.findall(rf'[gG]{{{i}}}', sequence, overlapped=True)) * 10 * i
        C_score += len(regex.findall(rf'[cC]{{{i}}}', sequence, overlapped=True)) * 10 * i
    if C_score == 0:
        C_score = 1
    return float(G_score) / float(C_score)


def g4hunter(sequence, output_map=False):
    """
    Apply a G4Hunter score calculation to the sequence.

    Default output is a mean value for the sequence, it should be used to
    analyze a sequence as whole.
    Map output is available providing G4H values for each nt, it allows the G/Cs
    at start or end of a window to have a value which takes into account the
    possible flanking G/Cs since they would have an impact on the G4H score at
    those positions.

    Returns a float between -4 and 4. [DEFAULT]
    Returns a list of floats between -4 and 4. [output_map=True]
    """
    g4h_list = [gr.group(1) for gr in regex.finditer(r'(?i)((?P<nt>[a-z])(?P=nt)*)',
                                                     sequence, overlapped=False)]
    g4h_map = [
        float(value) if -4 <= value <= 4 else (4.0 if value > 4 else -4.0)
        for match in [
            [len(x)] * len(x) if x[0].upper() == 'G'
            else ([-len(x)] * len(x) if x[0].upper() == 'C' else [0] * len(x))
            for x in g4h_list
        ]
        for value in match
    ]
    if output_map:
        return g4h_map
    else:
        return sum(g4h_map) / float(len(g4h_map))


def gen_G4RNA_df(
        seq_dict,
        columns,
        first_id,
        window_fragment=False,
        window_step=False,
        verbose=False):
    """
    Generate a dataframe like the one used to train the ANN from G4RNA to
    eventually concatenate them usually to G4RNA data. The dataframe can be
    filled with any controls (RNomic or scrambled) provided in a dictionary.

    When window_fragment and window_step are defined, the dataframe is
    populated with substring from dictionary instead of whole sequences.

    Return pandas dataframe.
    """
    data = []
    for key in seq_dict.keys():
        infos = utils.format_description(key)
        content = {ke: infos[ke] for ke in infos.keys()}
        if any(col in columns for col in [
            "gene_symbol", "mrnaAcc", "protAcc", "gene_stable_id",
            "transcript_stable_id", "full_name", "HGNC_id"
        ]):
            try:
                content.update(dict(zip(
                    ["gene_stable_id", "transcript_stable_id", "mrnaAcc", "gene_id",
                     "transcript_id", "gene_symbol", "gene_description"],
                    utils.retrieve_xref_Ensembl(
                        infos.get('stable_id'),
                        infos.get('mrnaAcc'),
                        utils.retrieve_RefSeq(infos.get('mrnaAcc'), infos.get('protAcc'))[2]
                    )
                )))
                content.update(dict(zip(
                    ["full_name", "HGNC_id"],
                    regex.search(
                        r'(?<full_name>.*) \[Source:HGNC Symbol;Acc:HGNC:(?<HGNC_id>\d+)\]',
                        content["gene_description"]
                    ).group('full_name', 'HGNC_id')
                )))
            except Exception:
                if 'stable_id' in infos and infos.get('stable_id'):
                    if infos.get('stable_id')[3] == 'T':
                        content['transcript_stable_id'] = infos.get('stable_id')
                    elif infos.get('stable_id')[3] == 'G':
                        content['gene_stable_id'] = infos.get('stable_id')
            try:
                content.update(dict(zip(
                    ["mrnaAcc", "protAcc", "gene_symbol", "product"],
                    utils.retrieve_RefSeq(infos["mrnaAcc"], infos["protAcc"])
                )))
            except Exception:
                pass
        for ke in infos.keys():
            if infos[ke]:
                content[ke] = infos[ke]
        if not window_fragment or window_fragment >= len(seq_dict[key]):
            seq = [seq_dict[key]]
        else:
            seq = [seq_dict[key][w:w + window_fragment] for w in range(
                0, len(seq_dict[key]) - window_fragment + window_step, window_step)]
            g4h_map = g4hunter(seq_dict[key], output_map=True)
        try:
            strt = int(infos.get('start'))
            nd = int(infos.get('end'))
            strnd = str(infos.get('strand'))
        except (TypeError, ValueError):
            strt = nd = strnd = None
        for no, s in enumerate(seq):
            row = []
            if strt is not None and strnd == '+':
                content['start'] = int(strt + no * window_step)
                content['end'] = int(strt + no * window_step + len(s) - 1)
            elif strt is not None and strnd == '-':
                content['start'] = int(nd - no * window_step - len(s) + 1)
                content['end'] = int(nd - no * window_step)
            else:
                content['start'] = int(no * window_step + 1)
                content['end'] = int(no * window_step + len(s))
            content['length'] = len(s)
            content['sequence'] = s
            content['cGcC'] = cgcc_scorer(s)
            if not window_fragment or window_fragment >= len(seq_dict[key]):
                content['G4H'] = g4hunter(s)
            else:
                content['G4H'] = sum(g4h_map[
                    no * window_step:no * window_step + window_fragment]) / float(len(
                    g4h_map[no * window_step:no * window_step + window_fragment]))
            content['g4'] = 'N/A'
            for column in columns:
                row.append(content.get(column, None))
            data.append(row)
    utils.verbosify(verbose, "DataFrame built")
    return pd.DataFrame(data=data, index=range(
        first_id, first_id + len(data)), columns=columns)


def submit_seq(
        ann,
        df_,
        except_columns,
        score_name,
        verbose=False):
    """
    Submit sequences from a dataframe to an ANN to be scored from 0 to 1.

    Return pandas dataframe with a new column "score_name".
    """
    trans_set_tst = df.drop(except_columns, axis=1)
    targets = [1.0 if str(df.loc[n].g4).lower() in ['yes', 'true'] else 0.0 for n in df.index]
    one_hot_encoder = OneHotEncoder(sparse=False)
    targets_one_hot = one_hot_encoder.fit_transform([[t] for t in targets])
    test_results = ann.predict(trans_set_tst.values)
    test_results = ann.activateOnDataset(alldata_tst)
    final_df = df_.copy()
    final_df[score_name] = pd.Series(test_results[:, 1], index=final_df.index)
    utils.verbosify(verbose, 'Sequence submitted')
    return final_df.drop(
        [c for c in final_df.columns if c not in except_columns][:-1],
        axis=1
    )