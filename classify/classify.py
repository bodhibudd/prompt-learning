import os, sys, logging
from transformers import Trainer, BertTokenizer, BertForMaskedLM, TrainingArguments,EarlyStoppingCallback
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import jieba
bert_model_path = "data/models/"
#可以进行扩展映射关系，比如好，美，可等，这里简单定义两个
ids2label = {"1":"好", "0":"差"}

train_path = os.path.join("data", "train_data.csv")
dev_path = os.path.join("data", "dev_data.csv")

data_dict = {"train": train_path, "dev": dev_path}

tokenizer = BertTokenizer.from_pretrained(bert_model_path)
model = BertForMaskedLM.from_pretrained(bert_model_path)

df_train = pd.read_csv(data_dict["train"], encoding="utf-8")
df_dev = pd.read_csv(data_dict["dev"], encoding="utf-8")

class ClassifyDataset(Dataset):
    def __init__(self, datas, max_len, labels=None):
        self.datas = datas
        self.max_len = max_len
        self.labels = labels
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        if self.labels is None:
            return (self.datas[idx],)
        else:
            return (self.datas[idx], self.labels[idx])

def default_collate_fn(examples):
    #数据进行长度统一
    #1表示训练与验证 1表示测试
    texts = []
    labels = []
    if len(examples[0]) == 2:
        for text, label in examples:
            texts.append(text)
            labels.append(label)
        length = max(len(text) for text in texts)
        max_len = length if length < 510 else 510
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_len)
        labels = tokenizer(labels, padding=True, truncation=True, max_length=max_len)
        label_ids = []
        for input_id, label_id in zip(inputs["input_ids"], labels["input_ids"]):
            assert len(label_id) == len(input_id)
            label_ids.append([id2 if id2 !=id1 else -100 for id1, id2 in zip(input_id, label_id)])
        inputs["labels"] = torch.tensor(label_ids, dtype=torch.long)
        inputs["input_ids"] = torch.tensor(inputs["input_ids"], dtype=torch.long)
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        inputs["token_type_ids"] = torch.tensor(inputs["token_type_ids"], dtype=torch.long)

        return inputs
    else:
        for text, label in examples:
            texts.append(text)
            labels.append(label)
        length = max(len(text) for text in texts)
        max_len = length if length < 510 else 510
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_len)
        inputs["input_ids"] = torch.tensor(inputs["input_ids"], dtype=torch.long)
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        inputs["token_type_ids"] = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
        return inputs

def get_dataloader(collate_fn, dataset, batch_size=4):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)


def get_datas(df):
    texts = []
    labels = []
    for index, row in tqdm(iterable=df.iterrows(), total=df.shape[0]):
        sent = row["review"]
        text = sent + "，酒店真[MASK]"
        label = sent + "，酒店真" + ids2label[str(row["label"])]
        texts.append(text)
        labels.append(label)
    return texts, labels


def train():
    t_texts, t_labels = get_datas(df_train)
    train_dataset = ClassifyDataset(t_texts, max_len=512, labels=t_labels)
    d_texts, d_labels = get_datas(df_dev)
    dev_dataset = ClassifyDataset(d_texts, max_len=512, labels=d_labels)
    args = TrainingArguments(output_dir="output/", evaluation_strategy="steps",
                             overwrite_output_dir=True,
                             num_train_epochs=2,
                             per_device_train_batch_size=4,
                             save_total_limit=3,
                             metric_for_best_model='loss',
                             greater_is_better=False,
                             load_best_model_at_end=True,
                             prediction_loss_only=True,
                             report_to="none")

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, data_collator=default_collate_fn)
    trainer.train()
    trainer.save_model(f'output/models')
def evaluate():
    model_path = "output/models/"
    d_texts, d_labels = get_datas(df_dev)
    dev_dataset = ClassifyDataset(d_texts, max_len=512, labels=d_labels)
    model = BertForMaskedLM.from_pretrained(model_path)
    total = 0

    for i, batch in tqdm(enumerate(get_dataloader(default_collate_fn, dev_dataset, 4))):

        encoder = model(**batch)
        scores = encoder.logits
        indexs = torch.argmax(scores, dim=2)
        index = torch.argmax(batch["labels"], dim=1)
        num = 0
        for j, idx in enumerate(index):
            if indexs[j,idx] != batch["labels"][j,idx]:
                print()
            num += (indexs[j,idx] == batch["labels"][j,idx])

        total += num
    acc = total/(1.0*len(dev_dataset))
    print("acc:{}".format(total/(1.0*len(dev_dataset))))
evaluate()



