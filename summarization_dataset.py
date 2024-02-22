import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, load_metric


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

class zhihu(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False): 
        super().__init__()
        if type_path=='train':
            self.dataset = load_dataset("arrow", data_files={'/scratch/dataset/Zhihu-KOL/train/data-00000-of-00005.arrow',
                                                          '/scratch/dataset/Zhihu-KOL/train/data-00001-of-00005.arrow',
                                                          '/scratch/dataset/Zhihu-KOL/train/data-00002-of-00005.arrow',
                                                          '/scratch/dataset/Zhihu-KOL/train/data-00003-of-00005.arrow'})
        else:
            self.dataset = load_dataset("arrow", data_files={'/scratch/dataset/Zhihu-KOL/train/data-00004-of-00005.arrow'})        
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
  
    def __len__(self):
        return self.dataset.shape[0]
    
    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')
        
        return text

    def __getitem__(self, index):
        dataset=self.dataset
        prompt, response = dataset[index]['INSTRUCTION'], dataset[index]['RESPONSE']
        source = self.tokenizer.batch_encode_plus([prompt], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([response], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
        #max_seq_len = self.max_seq_len - 5 # len('[EOS]') = 5
        # add an eos token note that end of resopnse, using in generate.
        #return f"{prompt[0: max_seq_len]}[EOS]", f"{response[0: max_seq_len]}[EOS]"

    """
    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['text']))
#         input_ = self.clean_text(example_batch['text']) + " </s>"
#         target_ = self.clean_text(example_batch['headline']) + " </s>"
        
        input_ = self.clean_text(example_batch['text'])
        target_ = self.clean_text(example_batch['headline'])
        
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
    
       
        return source, targets
  
    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
    """   
def get_dataset(tokenizer, type_path, num_samples, args):
      return zhihu(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=max_input_length, 
                        output_length=max_output_length)
