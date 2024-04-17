import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from transformers import TrainerCallback
global a

class LogPerformanceCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # 这里调用您的log_performance函数
        log_performance()
        test_results = trainer.evaluate(test_dataset)
        vali_results = trainer.evaluate(valid_dataset)
        print((test_results))
        print((vali_results))


def save_plot(performance, title, metric_name):
    epochs = list(range(1, len(performance) + 1))
    metric_values = [p[f'eval_{metric_name}'] for p in performance]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metric_values, label=f'{metric_name} Score')
    plt.title(f'{title} {metric_name} Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./{title.replace(" ", "_")}_{metric_name}.png')
    plt.close()

def log_performance():
    global test_performance, valid_performance
    test_eval = trainer.evaluate(test_dataset)
    valid_eval = trainer.evaluate(valid_dataset)
    test_performance.append(test_eval)
    valid_performance.append(valid_eval)
    save_plot(test_performance, 'Test Set', 'accuracy')
    save_plot(test_performance, 'Test Set', 'recall')
    save_plot(test_performance, 'Test Set', 'f1')
    save_plot(valid_performance, 'Validation Set', 'accuracy')
    save_plot(valid_performance, 'Validation Set', 'recall')
    save_plot(valid_performance, 'Validation Set', 'f1')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'recall': recall_score(labels, predictions, average='macro'),
        'f1': f1_score(labels, predictions, average='macro')
    }

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.label_dict = {label: idx for idx, label in enumerate(set(labels))}
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label_dict[self.labels[idx]]

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }


texts = []
labels = []
lines = open('../../data/goemotions_3.txt', encoding="utf-8").read().strip().split('\n')
for line in lines:
    last_comma_index = line.rfind(',')
    if last_comma_index != -1:
        input_text = line[:last_comma_index].strip('"')
        output_text = line[last_comma_index+1:].strip('"')
        texts.append(input_text)
        labels.append(output_text)


pretrained_model = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
full_dataset = TextDataset(texts, labels, tokenizer)
train_size = int(0.75 * len(full_dataset))
test_size = int(0.20 * len(full_dataset))
valid_size = len(full_dataset) - train_size - test_size

from transformers import BertConfig, BertModel

# 创建一个较小的BERT配置
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=len(set(labels)), hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024)
model = BertForSequenceClassification(config)


train_dataset, test_dataset, valid_dataset = random_split(full_dataset, [train_size, test_size, valid_size])
#model = AutoModelForSequenceClassification.from_pretrained('albert-base-v2', num_labels=len(set(labels)))
training_args = TrainingArguments(
    output_dir='../../bert_model',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=20,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2
)
test_performance = []
valid_performance = []

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    callbacks=[LogPerformanceCallback()]  # 添加自定义的回调
)

trainer.train()
test_results = trainer.evaluate(test_dataset)
vali_results = trainer.evaluate(valid_dataset)
print(test_results)
print(vali_results)

