### Nooshin ###
### Bert-RAI::test ###
## 03.11.2022##

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
import sys
data = sys.argv[1]
coi = sys.argv[2]

os.system("pip3 install transformers --use-feature=2020-resolver")
os.system("pip3 install torch")

import pandas as pd
import numpy as np
import torch
import random
from sklearn.model_selection import KFold
from torch.utils.data import random_split, SubsetRandomSampler
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import  confusion_matrix
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("finetuned_model/fu/model")

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

pd.set_option('mode.chained_assignment', None)
os.system("pip3 install openpyxl")
import openpyxl
df_raw = pd.read_excel(data)
df_raw['Annotation'] = 0
df_raw = df_raw.rename(columns={coi: "ReportText"})
df = df_raw[["ReportText","Annotation"]]
len(df)

df['ReportText']=df['ReportText'].astype(str)

df.head()

matches = ["see finding", "see above", "per narrative",
           "see below"]


for i, row in df.iterrows():
    modified_text_train = row["ReportText"].lower().\
    replace("\r"," ").replace("\n"," ").\
    replace("end of impression","").\
    replace("end impression","").\
    replace("end of  impression","").\
    replace("\t"," ")
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

for i, row in df.iterrows():
    modified_addendum_train = row["ReportText"].lower().\
    replace("\r"," ").replace("\n"," ").\
    replace("\t"," ")
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

df_new = df[["Annotation", "Modified_Text_new"]]
df_new = df_new.rename(columns={"Modified_Text_new": "Modified_Text"})

print(len(df_new['Annotation']))
print(len(df_new[df_new.Annotation == 0]))
print(len(df_new[df_new.Annotation == 1]))

df_new.index

#k_num = 5;
i=int(-1);

df_new['data_type'] = 'val'


encoded_data_val = tokenizer.batch_encode_plus(
  df_new[df_new.data_type == 'val'].Modified_Text.values,
  add_special_tokens = True,
  return_attention_mask = True,
  pad_to_max_length = True,
  truncation=True,
  max_length = 512,
  return_tensors = 'pt'
)


input_ids_val = encoded_data_val['input_ids']
attention_mask_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df_new[df_new.data_type == 'val'].Annotation.values)


dataset_val = TensorDataset(input_ids_val,
                            attention_mask_val, labels_val)
batch_size = 1

dataloader_val = DataLoader(
    dataset_val,
    sampler = None,
    batch_size = batch_size
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


val_loss, predictions, true_vals = evaluate(dataloader_val)
predictionsflat = np.argmax(predictions, axis=1).flatten()
true_valsflat = true_vals.flatten()

df_raw["RAI"]= predictionsflat
#df_raw['Modified_Text'] = df_new.loc[df_new['data_type'] =='val','Modified_Text'].values
df_raw = df_raw.drop(['Annotation'], axis =1)

outname = data[:-5]+'_RAI.xlsx'
df_raw.to_excel(outname)

