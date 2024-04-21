# -*- coding: utf-8 -*-
import pandas as pd
dataset_path = 'split_dataset/lem_train.csv'
# data = pd.read_csv(dataset_path)
# data = data.dropna()
# data.to_csv('preproc.csv')

import datasets
train_data = datasets.load_dataset("csv", data_files=dataset_path, split="train")

print(train_data)

train_data.info

import pandas as pd
from IPython.display import display, HTML
from datasets import ClassLabel

df = pd.DataFrame(train_data[:1])
for column, typ in train_data.features.items():
      if isinstance(typ, ClassLabel):
          df[column] = df[column].transform(lambda i: typ.names[i])
display(HTML(df.to_html()))

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-128k-uncased", strip_accents=False)

def map_to_length(x):
  x["text_len"] = len(tokenizer(x["text"]).input_ids)
  x["text_longer_512"] = int(x["text_len"] > 512)
  x["keywords_len"] = len(tokenizer(x["keywords"]).input_ids)
  x["keywords_longer_64"] = int(x["keywords_len"] > 64)
  x["keywords_longer_128"] = int(x["keywords_len"] > 128)
  return x

# sample_size = 10000
# data_stats = train_data.select(range(sample_size)).map(map_to_length, num_proc=4)

# def compute_and_print_stats(x):
#   if len(x["text_len"]) == sample_size:
#     print(
#         "Text Mean: {}, %-Texts > 512:{}, keywords Mean:{}, %-Keywords > 64:{}, %-Keywords > 128:{}".format(
#             sum(x["text_len"]) / sample_size,
#             sum(x["text_longer_512"]) / sample_size, 
#             sum(x["keywords_len"]) / sample_size,
#             sum(x["keywords_longer_64"]) / sample_size,
#             sum(x["keywords_longer_128"]) / sample_size,
#         )
#     )

# output = data_stats.map(
#   compute_and_print_stats, 
#   batched=True,
#   batch_size=-1,
# )

encoder_max_length=512
decoder_max_length=128

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["keywords"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

# train_data = train_data.select(range(32))

batch_size = 16
# batch_size=4

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["Unnamed: 0", "text", "keywords"]
)

train_data

train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)

val_data = datasets.load_dataset("csv", data_files='split_dataset/lem_val.csv', split="train")

val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["text", "keywords"]
)

val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)

from transformers import EncoderDecoderModel

bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("dbmdz/bert-base-turkish-128k-uncased", "dbmdz/bert-base-turkish-128k-uncased")

bert2bert

bert2bert.save_pretrained("bert2bert")

bert2bert = EncoderDecoderModel.from_pretrained("bert2bert")

bert2bert.config

bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

bert2bert.config.max_length = 142
bert2bert.config.min_length = 4
bert2bert.config.no_repeat_ngram_size = 2
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True, 
    output_dir="./",
    # logging_steps=2,
    # save_steps=10,
    # eval_steps=10,
    logging_steps=1000,
    save_steps=500,
    eval_steps=7500,
    warmup_steps=2000,
    # num_train_epochs=10,
    save_total_limit=3,
)

rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

print(f'STARTED TRAINING')
trainer = Seq2SeqTrainer(
    model=bert2bert,
    tokenizer=tokenizer,
    args=training_args,
    # compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()

# dummy_bert2bert = EncoderDecoderModel.from_pretrained("./checkpoint-20")

# from transformers import BertTokenizer

# bert2bert = dummy_bert2bert.to("cuda")
# tokenizer = BertTokenizer.from_pretrained("./checkpoint-20")

# test_data = datasets.load_dataset("csv", data_files=dataset_path, split="train[:1%]")

# def generate_summary(batch):
#     # cut off at BERT max length 512
#     inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
#     input_ids = inputs.input_ids.to("cuda")
#     attention_mask = inputs.attention_mask.to("cuda")

#     outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)

#     output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#     batch["pred_keywords"] = output_str

#     return batch

# batch_size = 16  # change to 64 for full evaluation

# results = test_data.select(range(3000)).map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["text"])

# results['pred_keywords']

# rouge.compute(predictions=results["pred_keywords"], references=results["keywords"], rouge_types=["rouge2"])["rouge2"].mid