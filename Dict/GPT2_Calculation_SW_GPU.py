# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:41:04 2024

@author: KX S
"""

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def sliding_window_information_content_entropy(text, model, tokenizer, max_length, stride):
    # Tokenize the text into characters
    characters = tokenizer.tokenize(text)
    print(len(characters))
    
    # Sliding window segmentation
    encoded_texts = []
    start = 0
    while start < len(characters):
        end = min(start + max_length, len(characters))
        encoded_text = tokenizer(
            characters[start:end], 
            return_tensors='pt', 
            is_split_into_words=True,
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )
        
        # Move the encoded text to the GPU
        encoded_text = {key: val.to(model.device) for key, val in encoded_text.items()}
        encoded_texts.append(encoded_text)
        if end == len(characters):
            break
        start += stride

    # Calculate information content and conditional entropy
    information_content_all = np.zeros(max_length + (len(characters) // stride) * stride)
    conditional_entropy_all = np.zeros(max_length + (len(characters) // stride) * stride)

    for i, encoded_text in enumerate(encoded_texts):
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']  # Use attention_mask

        input_ids_known = input_ids[:, :-1]
        target_ids = input_ids[:, 1:]

        with torch.no_grad():
            # Note: Pass attention_mask
            outputs = model(input_ids=input_ids_known, attention_mask=attention_mask[:, :-1])
            logits = outputs.logits

        probs = F.softmax(logits, dim=-1)

        # Calculate information content
        log_probs_list = []
        information_content_list = [0] * target_ids.size(1)
        
        for t in range(target_ids.size(1)):
            target_prob = probs[:, t, :].gather(1, target_ids[:, t].unsqueeze(-1))  # Calculate conditional probability of each token, given all the tokens before the target token
            log_prob = torch.log2(target_prob)  # Compute log(p)
            log_probs_list.append(log_prob.item())  # Append log(p) to the list
            information_content_list[t] = log_prob.item()  # Compute information content = log(p) and accumulate
            
        print("information_content_list", len(information_content_list))
        print("information_content_all", len(information_content_all))
     
        
        # Calculate conditional entropy
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Merge results
        start_idx = i * stride
        if i == 0:
            # For the first window, calculate information content and conditional entropy for all characters
            print("start_idx", start_idx)
            print("start_idx + len(information_content_list)", start_idx + len(information_content_list))
            information_content_all[start_idx : start_idx + len(information_content_list)] = np.array(information_content_list)
            conditional_entropy_all[start_idx : start_idx + entropy.size(1)] = entropy.cpu().numpy().flatten()
        else:
            # For subsequent windows, calculate information content and conditional entropy only from the 256th character onward
            overlap_start = stride
            information_content_all[start_idx + overlap_start : start_idx + len(information_content_list)] = np.array(information_content_list)[overlap_start:]
            conditional_entropy_all[start_idx + overlap_start : start_idx + entropy.size(1)] = entropy[:, overlap_start:].cpu().numpy().flatten()
    
    # Discard the excess portion
    information_content_all = information_content_all[0:len(characters)]
    conditional_entropy_all = conditional_entropy_all[0:len(characters)]
    print("information_content_all", len(information_content_all))

    return information_content_all, conditional_entropy_all






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = GPT2Tokenizer.from_pretrained('/content/drive/MyDrive/Colab Notebooks/results')
tokenizer.pad_token = tokenizer.eos_token  
model = GPT2LMHeadModel.from_pretrained('/content/drive/MyDrive/Colab Notebooks/results').to(device)