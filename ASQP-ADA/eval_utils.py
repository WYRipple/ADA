# -*- coding: utf-8 -*-

import re
from data_utils import aspect_cate_list
import pdb

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}
aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'food general',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']

def extract_spans_para(task, seq, seq_type,use_new_target):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    if task == 'asqp':
        for s in sents:
            try:
                if use_new_target == 1:
                    ac_at, ot_sp = s.split(" is ")
                    ac, at = ac_at.split(" of ")
                    if "and" in ot_sp:
                        ot, sp = ot_sp.split(" and ")
                    else:
                        sp = ot_sp
                        ot = "NULL"
                    if at == "something":
                        at = "NULL"
                else:
                    ac_sp, at_ot = s.split(' because ')
                    ac, sp = ac_sp.split(' is ')
                    at, ot = at_ot.split(' is ')
                    if at.lower() == 'it':
                        at = 'NULL'
            except ValueError:
                try:
                    pass
                except UnicodeEncodeError:
                    pass
                ac, at, sp, ot = '', '', '', ''

            quads.append((ac, at, sp, ot))
    else:
        raise NotImplementedError
    return quads


def compute_f1_scores(pred_pt, gold_pt, caculate_cate):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    n_tp, n_gold, n_pred = 0, 0, 0
    for i in range(len(pred_pt)):
        pred_pt_set = set(pred_pt[i])
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt_set)

        for t in pred_pt_set:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}
    print(scores)

    if caculate_cate == 1:
        for cate in aspect_cate_list:
            n_tp, n_gold, n_pred = 0, 0, 0
            for i in range(len(pred_pt)):
                pred_pt_set = set(pred_pt[i])
                cate_pred_pt_set = []
                cate_gold_pt = []
                for label in pred_pt_set:
                    if label[0] == cate:
                        cate_pred_pt_set.append(label)
                for label in gold_pt[i]:
                    if label[0] == cate:
                        cate_gold_pt.append(label)            
                n_gold += len(cate_gold_pt)
                n_pred += len(cate_pred_pt_set)
                for t in cate_pred_pt_set:
                    if t in cate_gold_pt:
                        n_tp += 1
            print(f"number of gold spans for {cate}: {n_gold}, predicted spans for {cate}: {n_pred}, hit: {n_tp}")
            precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
            recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
            scores = {'precision': precision, 'recall': recall, 'f1': f1}
            print(scores)

    return scores


def compute_scores(pred_seqs, gold_seqs, use_new_target, caculate_cate):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []
    for i in range(num_samples):
        gold_list = extract_spans_para('asqp', gold_seqs[i], 'gold',use_new_target)
        pred_list = extract_spans_para('asqp', pred_seqs[i], 'pred',use_new_target)
        all_labels.append(gold_list)
        all_preds.append(pred_list)
    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels, caculate_cate)
    return scores, all_labels, all_preds
