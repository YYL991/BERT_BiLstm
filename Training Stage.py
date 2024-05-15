import shutil
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics.functional import accuracy, recall, precision, f1_score 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

#copy data
shutil.copytree("/content/drive/MyDrive/data/sst","/content/sst")

# Define datasets
class MydataSet(Dataset):
    def __init__(self, path, split):
        self.dataset = load_dataset('csv', data_files=path, split=split)

    def __getitem__(self, item):
        text = self.dataset[item]['text']
        label = self.dataset[item]['label']
        return text, label

    def __len__(self):
        return len(self.dataset)


# Define batch functions
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,  # Truncate when sentence length is greater than max_length.
        padding='max_length',  # pad to max_length
        max_length=200,
        return_tensors='pt',  # 
        return_length=True,
    )

    input_ids = data['input_ids']   # input_ids are the encoded words.
    attention_mask = data['attention_mask']  # The pad position is 0, other positions are 1.
    token_type_ids = data['token_type_ids'] # the position of the first sentence and the special symbol is 0, the position of the second sentence is 1
    labels = torch.LongTensor(labels)  # labels for the batch

    return input_ids, attention_mask, token_type_ids, labels



#Define the model, use bert pre-training upstream, choose a bi-directional LSTM model for the downstream task, and finally add a fully connected layer
class BiLSTMClassifier(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim, extra_hidden_dim):
        super(BiLSTMClassifier, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.extra_hidden_dim = extra_hidden_dim

        # load BERT model, generate word embeddings
        self.embedding = BertModel.from_pretrained('bert-base-uncased')
        # Freeze upstream model parameters (no pre-training model parameter learning)
        for param in self.embedding.parameters():
            param.requires_grad_(False)
        # Generate downstream RNN layers as well as fully connected layers
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=self.drop)
        self.extra_fc = nn.Linear(self.hidden_dim * 2, self.extra_hidden_dim)
        self.fc = nn.Linear(self.extra_hidden_dim, self.output_dim)
        #self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)


    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedded = embedded.last_hidden_state  
        out, (h_n, c_n) = self.lstm(embedded)
        output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = torch.relu(self.extra_fc(output))
        output = self.fc(output)
        log_probs = F.log_softmax(output)#dim=1？
        return log_probs,output





# define pytorch lightning
class BiLSTMLighting(pl.LightningModule):
    def __init__(self, drop, hidden_dim, output_dim,extra_hidden_dim):
        super(BiLSTMLighting, self).__init__()
        self.model = BiLSTMClassifier(drop, hidden_dim, output_dim, extra_hidden_dim)  # Setting up the model
        self.criterion = nn.NLLLoss()  # Setting the loss function

        self.train_dataset = MydataSet('/content/sst/train2.csv', 'train')
        self.val_dataset = MydataSet('/content/sst/valid2.csv', 'train')
        self.test_dataset = MydataSet('/content/sst/train2.csv', 'train')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    def forward(self, input_ids, attention_mask, token_type_ids):  # forward(self,x)
        return self.model(input_ids, attention_mask, token_type_ids)

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                  shuffle=True)
        return train_loader

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        y = labels
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  
        return loss

    def val_dataloader(self):
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return val_loader

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        y = labels                                                         
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        target = labels

        self.log('val_loss', loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        #self.log('val_acc', accuracy(pred, target, task="binary", num_classes=class_num))
        self.log('val_acc', accuracy(pred, target, task="binary", num_classes=class_num),
                prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return test_loader

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels  = batch
        target = labels  
        y = labels
        y_hat,result = self.model(input_ids, attention_mask, token_type_ids)
        numpy_array1 = y_hat.cpu().numpy()
        numpy_array2 = result.cpu().numpy()
        with open('./result/tensor_train.csv','a') as f:
          np.savetxt(f, numpy_array2, delimiter=',')
        pred = torch.argmax(y_hat, dim=1)

        loss = self.criterion(y_hat, y)
        self.log('loss', loss)

        self.log('acc', accuracy(pred, target, task="binary", num_classes=class_num))
        self.log('recall', recall(pred, target, task="binary", num_classes=class_num, average="weighted"))
        self.log('precision', precision(pred, target, task="binary", num_classes=class_num, average="weighted"))
        self.log('f1', f1_score(pred, target, task="binary", num_classes=class_num, average="weighted"))






def test():
    # Load previously trained optimal model parameters
    model = BiLSTMLighting.load_from_checkpoint(checkpoint_path=PATH,drop=dropout, hidden_dim=hidden_dim, output_dim=class_num,extra_hidden_dim=extra_hidden_dim)
    trainer = Trainer(fast_dev_run=False)
    result = trainer.test(model)
    print(result)



def train():
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # monitor object as 'val_loss'
        dirpath='checkpoints/',  # Path to save the model
        filename='model-{epoch:02d}-{val_loss:.2f}',  # Name of the optimal model
        save_top_k=1,  # Save only the best one
        mode='min'  # When the monitoring object indicator is the smallest
    )
    

    trainer = Trainer(max_epochs=epochs, log_every_n_steps=10, accelerator='gpu', devices="auto", fast_dev_run=False,
                      precision=16, callbacks=[checkpoint_callback])
    model = BiLSTMLighting(drop=dropout, hidden_dim=hidden_dim, output_dim=class_num, extra_hidden_dim = extra_hidden_dim)
    trainer.fit(model)









if __name__ == '__main__':
    #define hyperparameters
    batch_size = 256
    epochs = 10
    dropout = 0.5
    hidden_dim = 768
    extra_hidden_dim= 256
    class_num = 2
    lr = 0.001

    PATH = '/content/checkpoints/model-epoch=04-val_loss=0.24.ckpt'
    token = BertTokenizer.from_pretrained('bert-base-uncased')
    train() #No.12
    #test()
