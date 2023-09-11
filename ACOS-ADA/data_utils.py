# -*- coding: utf-8 -*-

import random
from torch.utils.data import Dataset

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

rest_aspect_cate_list = ['location general',
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
laptop_aspect_cate_list = ['battery design_features', 'battery general', 'battery operation_performance', 'battery quality', 'company design_features', 'company general', 'company operation_performance', 'company price', 'company quality', 'cpu design_features', 'cpu general', 'cpu operation_performance', 'cpu price', 'cpu quality', 'display design_features', 'display general', 'display operation_performance', 'display price', 'display quality', 'display usability', 'fans&cooling design_features', 'fans&cooling general', 'fans&cooling operation_performance', 'fans&cooling quality', 'graphics design_features', 'graphics general', 'graphics operation_performance', 'graphics usability', 'hard_disc design_features', 'hard_disc general', 'hard_disc miscellaneous', 'hard_disc operation_performance', 'hard_disc price', 'hard_disc quality', 'hard_disc usability', 'hardware design_features', 'hardware general', 'hardware operation_performance', 'hardware price', 'hardware quality', 'hardware usability', 'keyboard design_features', 'keyboard general', 'keyboard miscellaneous', 'keyboard operation_performance', 'keyboard portability', 'keyboard price', 'keyboard quality', 'keyboard usability', 'laptop connectivity', 'laptop design_features', 'laptop general', 'laptop miscellaneous', 'laptop operation_performance', 'laptop portability', 'laptop price', 'laptop quality', 'laptop usability', 'memory design_features', 'memory general', 'memory operation_performance', 'memory quality', 'memory usability', 'motherboard general', 'motherboard operation_performance', 'motherboard quality', 'mouse design_features', 'mouse general', 'mouse usability', 'multimedia_devices connectivity', 'multimedia_devices design_features', 'multimedia_devices general', 'multimedia_devices operation_performance', 'multimedia_devices price', 'multimedia_devices quality', 'multimedia_devices usability', 'optical_drives design_features', 'optical_drives general', 'optical_drives operation_performance', 'optical_drives usability', 'os design_features', 'os general', 'os miscellaneous', 'os operation_performance', 'os price', 'os quality', 'os usability', 'out_of_scope design_features', 'out_of_scope general', 'out_of_scope operation_performance', 'out_of_scope usability', 'ports connectivity', 'ports design_features', 'ports general', 'ports operation_performance', 'ports portability', 'ports quality', 'ports usability', 'power_supply connectivity', 'power_supply design_features', 'power_supply general', 'power_supply operation_performance', 'power_supply quality', 'shipping general', 'shipping operation_performance', 'shipping price', 'shipping quality', 'software design_features', 'software general', 'software operation_performance', 'software portability', 'software price', 'software quality', 'software usability', 'support design_features', 'support general', 'support operation_performance', 'support price', 'support quality', 'warranty general', 'warranty quality']

def read_line_examples_from_file(data_path, data_dir, use_prompt):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    if data_dir == "laptop-acos":
        aspect_cate_list_to_sent = ','.join(laptop_aspect_cate_list)
    elif data_dir == "rest-acos":
        aspect_cate_list_to_sent = ','.join(rest_aspect_cate_list)
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

            man_ot = sentword2opinion[sp]  # 'POS' -> 'good'    
            man_ot = sentword2opinion[sp]  # 'POS' -> 'good'    
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
    sents, labels = read_line_examples_from_file(data_path, data_dir, use_prompt)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    task = 'asqp'
    if task == 'asqp':
        targets = get_para_asqp_targets(sents, labels, use_newtarget)
    else:
        raise NotImplementedError

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128, use_prompt=1, use_newtarget=1):
        # './data/rest16/train.txt'
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
