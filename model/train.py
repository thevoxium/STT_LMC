import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering, TrainingArguments, Trainer
import tensorflow as tf

data = pd.read_csv('data.csv')
dataset = Dataset.from_pandas(data)

dataset = dataset.train_test_split(test_size=0.1)


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['questions'], batch['answers'], padding='max_length', truncation=True)

train_dataset = dataset['train'].map(tokenize, batched=True, batch_size=len(dataset['train']))
val_dataset = dataset['test'].map(tokenize, batched=True, batch_size=len(dataset['test']))

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])


model = TFAutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


trainer.train()


def get_answer(question):
    inputs = tokenizer(question, return_tensors='tf')
    outputs = model(inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits


    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0] 
    answer_end = (tf.argmax(answer_end_scores, axis=1) + 1).numpy()[0] 


    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer
