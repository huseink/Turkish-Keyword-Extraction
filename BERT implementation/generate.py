#coding:utf8
from transformers import EncoderDecoderModel
import datasets
import pandas as pd
from nltk.tokenize import word_tokenize

dataset_path = 'test_new.csv'

checkpoint_path = 'succesful_model_second/checkpoint-20500'

dummy_bert2bert = EncoderDecoderModel.from_pretrained(checkpoint_path)

from transformers import BertTokenizer

bert2bert = dummy_bert2bert.to("cuda")
tokenizer = BertTokenizer.from_pretrained(checkpoint_path, strip_accents=False)

test_data = datasets.load_dataset("csv", data_files=dataset_path, split="train")

rouge = datasets.load_metric("rouge")
meteor = datasets.load_metric("meteor")
bleu = datasets.load_metric("bleu")
sacrebleu = datasets.load_metric("sacrebleu")
bertscore = datasets.load_metric("bertscore")

def generate_summary(batch):
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred_keywords"] = output_str

    return batch

batch_size = 16  # change to 64 for full evaluation

results = test_data.map(generate_summary, batched=True, batch_size=batch_size)

text = list(results['text'])
pred_keywords = list(results['pred_keywords'])
keywords = list(results['keywords'])

tokenized_predictions = [word_tokenize(pred) for pred in pred_keywords]
tokenized_references = [[word_tokenize(ref)] for ref in keywords]


score = rouge.compute(predictions=results['pred_keywords'], references=results['keywords'], rouge_types=["rouge1"])["rouge1"].mid
score_r2 = rouge.compute(predictions=results['pred_keywords'], references=results['keywords'], rouge_types=["rouge2"])["rouge2"].mid
meteor_score = meteor.compute(predictions=pred_keywords, references=keywords)
bleu_score = bleu.compute(predictions=tokenized_predictions, references=tokenized_references)
bert_score = bertscore.compute(predictions=results['pred_keywords'], references=results['keywords'], lang='tr')
sacrebleu_score = sacrebleu.compute(predictions=tokenized_predictions, references=tokenized_references)
bert_score_array = [round(v, 2) for v in bert_score["f1"]]
bert_score_average = sum(bert_score_array) / len(bert_score_array)
rounded_bert_score_average = round(bert_score_average, 3)
# Define the text file path
txt_file_path = 'generate-extra-makale-koksuz.txt'

# Open the text file in write mode
with open(txt_file_path, 'w', encoding='utf-8') as file:
    file.write('--------------------------\n')
    file.write('Score Evaluations\n')
    file.write('Rouge-1: {}\n'.format(score))
    file.write('Rouge-2: {}\n'.format(score_r2))
    file.write('Meteor: {}\n'.format(round(meteor_score["meteor"], 3)))
    file.write('Bleu: {}\n'.format(round(bleu_score['bleu'],3)))
    file.write('SacreBleu: {}\n'.format(round(sacrebleu_score['score'], 3)))
    file.write('Bert: {}\n'.format(rounded_bert_score_average))
    file.write('--------------------------')
    for text, keyword, predicted_keyword in zip(text, keywords, pred_keywords):
        file.write('Makale: {}\n'.format(text))
        file.write('Anahtar sozcukler: {}\n'.format(keyword))
        file.write('Uretilen anahtar sozcukler: {}\n'.format(predicted_keyword))
        file.write('----------------------\n')

print('Text file saved successfully.')
