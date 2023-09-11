# -*- coding: utf-8 -*-

import random
from torch.utils.data import Dataset

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

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
aspect_cate_list_to_sent = ','.join(aspect_cate_list)

def read_line_examples_from_file(data_path, use_prompt):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                if use_prompt == 1:
                    words = words + aspect_cate_list_to_sent 
                sents.append(words.split())
                labels.append(eval(tuples))
    return sents, labels


def get_para_asqp_targets(sents, labels, use_newtarget):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad

            man_ot = sentword2opinion[sp]   
            if use_newtarget == 1:
                if at == "NULL":
                    at = 'something'
                if ot == "NULL":
                    one_quad_sentence = f"{ac} of {at} is {man_ot}"
                else:
                    one_quad_sentence = f"{ac} of {at} is {ot} and {man_ot}"
            else:
                if at == 'NULL':
                    at = 'it'
                one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"

            all_quad_sentences.append(one_quad_sentence)
        
        sentences = ' [SSEP] '.join(all_quad_sentences)
        targets.append(sentences)
    return targets


def get_transformed_io(data_path, data_dir, use_prompt, use_newtarget):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path, use_prompt)
    inputs = [s.copy() for s in sents]

    task = 'asqp'
    if task == 'asqp':
        targets = get_para_asqp_targets(sents, labels, use_newtarget)
    else:
        raise NotImplementedError

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128, use_prompt=1, use_newtarget=1):
        self.data_path = f'data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.use_prompt = use_prompt
        self.use_newtarget = use_newtarget

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze() 
        target_mask = self.targets[index]["attention_mask"].squeeze()  

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.data_dir, self.use_prompt, self.use_newtarget)

        for i in range(len(inputs)):
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
