# -*- coding: utf-8 -*-
"""BERT-based AI algorithme for
finding Recommendations for Adiitional Imaging (RAI) in radiology reports.
"""

# gpu_info = !nvidia-smi
# gpu_info = '\n'.join(gpu_info)
# if gpu_info.find('failed') >= 0:
#   print('Not connected to a GPU')
# else:
#   print(gpu_info)

import GPUtil
GPUtil.showUtilization()

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')

import os
os.system("pip3 install transformers --use-feature=2020-resolver")
os.system("pip3 install torch")

#!pip install transformers --use-feature=2020-resolver
#!pip install torch

import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import random_split, SubsetRandomSampler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import  confusion_matrix
from sklearn.model_selection import KFold

def pr_rc_f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_recall_fscore_support(labels_flat, preds_flat, average = 'weighted')

def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in tqdm(dataloader_val):
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    loss_val_avg = loss_val_total/len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals

from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
)

import pandas as pd
df_raw = pd.read_csv('trainingsets/fu/finalAnnotatedDataforNLP.csv')
# df_raw.head()

df = df_raw[["ReportText","Annotation"]]
len(df)


df.ReportText=df.ReportText.astype(str)

df.head()

#a_string = "A string is more than its parts!"
matches = ["see finding", "see above", "per narrative",
           "see below"]


for i, row in df.iterrows():
    modified_text_train = row["ReportText"].lower().\
    replace("\r"," ").replace("\n"," ").\
    replace("end of impression","").\
    replace("end impression","").\
    replace("end of  impression","")
    if any(x in modified_text_train for x in matches):
      df.at[i,'Modified_Text'] = modified_text_train.split("impression:")[0].\
      split("findings:")[-1]
    elif any("econsult"):
      df.at[i,'Modified_Text'] = modified_text_train.split("econsult answer:")[-1]
    else:
      df.at[i,'Modified_Text'] = modified_text_train.split("impression:")[-1].\
      split("interpretation summary")[-1].\
      split("conclusion:")[-1].\
      split("interpretation:")[-1].\
      split("findings:")[-1]

#print(row['Modified_Text'])

#df_new = df[["Annotation", "Modified_Text"]]
#df_new["Modified_Text"][6473]

for i, row in df.iterrows():
    modified_addendum_train = row["ReportText"].lower().\
    replace("\r"," ").replace("\n"," ")
    if "addendum:" in modified_addendum_train:
      if "addended" in modified_addendum_train:
        df.at[i,'addendum'] = modified_addendum_train.split("addendum:")[-1].\
        split("addended")[0]
      elif "end of addendum" in modified_addendum_train:
        df.at[i,'addendum'] = modified_addendum_train.split("addendum:")[-1].\
        split("end of addendum")[0]
      else:
        df.at[i,'addendum'] = modified_addendum_train.split("addendum:")[-1].\
        split("reason for exam")[0]


df['Modified_Text_new'] = df['Modified_Text'].map(str) + '-' + df['addendum'].map(str)

#df.head()

df_new = df[["Annotation", "Modified_Text_new"]]

df_new = df_new.rename(columns={"Modified_Text_new": "Modified_Text"})

#df_new.tail(2)

print(len(df_new['Annotation']))
print(len(df_new[df_new.Annotation == 0]))
print(len(df_new[df_new.Annotation == 1]))

df_new.index

k_num = 5;
i=int(-1);


train_idx = range(len(df_new['Annotation']))

df_new['data_type'] = ['not_set']*df_new.shape[0]
df_new.loc[train_idx, 'data_type'] = 'train'

encoded_data_train = tokenizer.batch_encode_plus(
  df_new[df_new.data_type == 'train'].Modified_Text.values,
  add_special_tokens = True,
  return_attention_mask = True,
  pad_to_max_length = True,
  max_length = 512,
  return_tensors = 'pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_mask_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df_new[df_new.data_type == 'train'].Annotation.values)

dataset_train = TensorDataset(input_ids_train,
                            attention_mask_train, labels_train)

model = BertForSequenceClassification.from_pretrained(
  'bert-base-uncased',
  num_labels = 2,
  output_attentions = False,
  output_hidden_states = False
)
batch_size = 1

dataloader_train = DataLoader(
    dataset_train,
    sampler = RandomSampler(dataset_train),
    batch_size = batch_size
)


optimizer = AdamW(
  model.parameters(),
  lr = 1e-5, #2e-5 > 5e-5
  eps = 1e-8
)
epochs = 5

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = len(dataloader_train)*epochs
)
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in tqdm(range(1, epochs+1)):
  i=i+1;
  model.train()
  loss_train_total = 0
    #progress_bar = tqdm(dataloader_train,
    #                   desc = 'Epoch {:id}'.format(epoch),
    #                   leave = False,
    #                   disable=False)
  progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
  for batch in progress_bar:
      model.zero_grad()
        #batch = tuple(b.to(device) for b in batch)
      batch = tuple(b.to(device) for b in batch)
      inputs = {
          'input_ids'           :batch[0],
          'attention_mask'      :batch[1],
          'labels'              :batch[2]
      }
      outputs = model(**inputs)
      loss = outputs[0]
        #loss_train_total += loss.item()
      loss_train_total += loss.item()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()
      progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
    #torch.save(model.state_dict(), f'Models/BERT_ft_epoch{epoch}.model')
  tqdm.write(f'\nEpoch: {epoch}')
  loss_train_avg = loss_train_total/len(dataloader_train)
  tqdm.write(f'Training loss: {loss_train_avg}')

range(len(df_raw['Annotation']),len(df_new['Annotation']))


# save model
model.save_pretrained("finetuned_model/fu/model")

