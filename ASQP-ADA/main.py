# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores

import random
import numpy as np
import pdb

logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='asqp', type=str, required=True,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    parser.add_argument("--use_category_prompt", default=1, type=int)
    parser.add_argument("--use_new_target", default=1, type=int)
    parser.add_argument("--save_name", default="save", type=str)
    parser.add_argument("--dataset_name", default="train", type=str)
    parser.add_argument("--caculate_cate", default=0, type=int)
    parser.add_argument("--caculate_pattern", default=0, type=int)

    args = parser.parse_args()
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    output_dir = f"outputs/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args

class ASQPDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.load_dataset()

    def load_dataset(self):
        train_dataset = ABSADataset(self.tokenizer, data_dir=args.dataset, data_type=args.dataset_name,
                                    max_len=args.max_seq_length,use_prompt=args.use_category_prompt, use_newtarget=args.use_new_target)
        dev_dataset = ABSADataset(self.tokenizer, data_dir=args.dataset, data_type='dev',
                                   max_len=args.max_seq_length,use_prompt=args.use_category_prompt, use_newtarget=args.use_new_target)
        test_dataset = ABSADataset(self.tokenizer, data_dir=args.dataset, data_type='test',
                                    max_len=args.max_seq_length,use_prompt=args.use_category_prompt, use_newtarget=args.use_new_target)
        self.raw_datasets = {
            'train': train_dataset,
            'dev': dev_dataset,
            'test': test_dataset
        }

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(dataset=self.raw_datasets[mode], batch_size=batch_size, num_workers=4, shuffle=shuffle)
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", 32, shuffle=False)

class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams, tfm_model, tokenizer, data_module):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = tfm_model
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.use_new_target = hparams.use_new_target
    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        f1 = evaluate(self.data_module.val_dataloader(), self, self.use_new_target, caculate_cate=0)
        self.current_val_result = f1["f1"]
        if not hasattr(self, 'best_val_result') or self.best_val_result is None:
            self.best_val_result = self.current_val_result
        elif self.best_val_result < self.current_val_result:
            self.best_val_result = self.current_val_result
            self.save_model()

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        self.lr_scheduler = scheduler
        return [optimizer]
    
    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = os.path.join(self.hparams.output_dir, str(self.hparams.save_name), 'model')
        print(f'\n## save model to {dir_name}\n')
        self.model.save_pretrained(dir_name)

    def load_model(self):
        dir_name = os.path.join(self.hparams.output_dir, str(self.hparams.save_name), 'model')
        print(f'\n## load model to {dir_name}\n')
        self.model = T5ForConditionalGeneration.from_pretrained(dir_name)


    def setup(self, stage):
        if stage == 'fit':
            self.train_loader = self.train_dataloader()
            ngpus = 1
            effective_batch_size = self.hparams.train_batch_size * self.hparams.gradient_accumulation_steps * ngpus
            dataset_size = len(self.train_loader.dataset)
            self.total_steps = (dataset_size / effective_batch_size) * self.hparams.num_train_epochs

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None):
        optimizer.step(closure=optimizer_closure) 
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        print("F1 : ", pl_module.current_val_result)
        print("Best F1 : ", pl_module.best_val_result)

def evaluate(data_loader, model, use_new_target, caculate_cate):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu}')
    model.model.to(device)

    model.model.eval()

    outputs, targets = [], []

    for batch in tqdm(data_loader):
        outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=128)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    '''
    print("\nPrint some results to check the sanity of generation method:", '\n', '-'*30)
    for i in [1, 5, 25, 42, 50]:
        try:
            print(f'>>Target    : {targets[i]}')
            print(f'>>Generation: {outputs[i]}')
        except UnicodeEncodeError:
            print('Unable to print due to the coding error')
    print()
    '''

    scores, all_labels, all_preds = compute_scores(outputs, targets, use_new_target, caculate_cate)
    results = {'scores': scores, 'labels': all_labels, 'preds': all_preds}

    return scores

args = init_args()
print("\n", "="*30, f"NEW EXP: ASQP on {args.dataset}", "="*30, "\n")
data_module = ASQPDataModule(args)
data_module.load_dataset()
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
print(f"Here is an example (from the dev set):")
dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                      data_type='dev', max_len=args.max_seq_length,use_prompt=args.use_category_prompt, use_newtarget=args.use_new_target)
data_sample = dataset[7]
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

if args.do_train:
    print("\n****** Conduct Training ******")
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = T5FineTuner(args, tfm_model, tokenizer, data_module)
    train_params = dict(
        default_root_dir=args.output_dir,
        checkpoint_callback=False,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        callbacks=[LoggingCallback()],
    )
    trainer = pl.Trainer(**train_params)
    trainer.fit(model, datamodule=data_module)
    print("Finish training and saving the model!")

if args.do_direct_eval:
    print("\n****** Conduct Evaluating with the last state ******")
    if args.do_train != True:
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model = T5FineTuner(args, tfm_model, tokenizer, data_module)
    model.load_model()
    sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt',use_prompt=args.use_category_prompt)
    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                               data_type='test', max_len=args.max_seq_length,use_prompt=args.use_category_prompt, use_newtarget=args.use_new_target)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    scores = evaluate(test_loader, model, args.use_new_target, args.caculate_cate)
    log_file_path = f"results_log/{args.dataset}.txt"
    local_time = time.asctime(time.localtime(time.time()))
    exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
    exp_results = f"F1 = {scores['f1']:.4f}"
    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    data_type_for_3 = ['test_style_1', "test_style_2", "test_style_3"]
    for data_type_choose in  data_type_for_3:
        txt_name = data_type_choose + ".txt"
        print(data_type_choose)
        sents, _ = read_line_examples_from_file(f'data/{args.dataset}/{txt_name}',use_prompt=args.use_category_prompt)
        test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                                data_type=data_type_choose, max_len=args.max_seq_length,use_prompt=args.use_category_prompt, use_newtarget=args.use_new_target)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
        scores = evaluate(test_loader, model, args.use_new_target,caculate_cate=0)
        log_file_path = f"results_log/{args.dataset}.txt"
        local_time = time.asctime(time.localtime(time.time()))
        exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
        exp_results = f"F1 = {scores['f1']:.4f}"
        log_str = f'============================================================\n'
        log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if args.caculate_pattern == 1:
        folder_path = "" + args.dataset
        data_type_for_manydag = os.listdir(folder_path)
        for data_type_choose in  data_type_for_manydag:
            print(data_type_choose)
            data_type_choose_wo_txt = data_type_choose[:-4]
            sents, _ = read_line_examples_from_file(f'data/{args.dataset}/{data_type_choose}',use_prompt=args.use_category_prompt)
            test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                                    data_type=data_type_choose_wo_txt, max_len=args.max_seq_length,use_prompt=args.use_category_prompt, use_newtarget=args.use_new_target)
            test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
            scores = evaluate(test_loader, model, args.use_new_target, caculate_cate=0)
            log_file_path = f"results_log/{args.dataset}.txt"
            local_time = time.asctime(time.localtime(time.time()))
            exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
            exp_results = f"F1 = {scores['f1']:.4f}"
            log_str = f'============================================================\n'
            log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"



