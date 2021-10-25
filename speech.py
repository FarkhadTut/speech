import torch
import torchaudio
import torch.nn as nn
import pandas as pd 
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from pytorch_lightning.core.lightning import LightningModule
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from glob import glob
import torch.optim as optim
import sys
from dataloader import train_dataloader, val_dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ActDropNormCNN1d(nn.Module):
	def __init__(self, 	n_feats, dropout, keep_shape=False):
		super(ActDropNormCNN1d, self).__init__()
		self.dropout = nn.Dropout(dropout)
		self.norm = nn.LayerNorm(n_feats)
		self.keep_shape = keep_shape
		self.gelu = nn.GELU()

	def forward(self, t):
		t = t.transpose(1,2)
		t = self.dropout(self.gelu(self.norm(t)))
		if self.keep_shape:
			return t.transpose(1,2) 
		return t


class SpeecRecognition(LightningModule):
	hyper_parameters = {
		'num_classes': 43,
		'n_feats': 81,
		'dropout': 0.1,
		'hidden_size': 1024,
		'num_layers': 2,
		'lr': 0.001,
	}

	def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout, lr=0.0001):
		super(SpeecRecognition, self).__init__()
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
		self.lr = lr
		self.current_val_loss = None
		self.cnn = nn.Sequential(
			nn.Conv1d(n_feats, n_feats, 2, padding=1),
			ActDropNormCNN1d(n_feats, dropout),

		)
		self.dense = nn.Sequential(
			nn.Linear(n_feats, 256),
			nn.LayerNorm(256),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(256, 256),
			nn.LayerNorm(256),
			nn.GELU(),
			nn.Dropout(dropout),

		)

		self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size,
							num_layers=num_layers, dropout=0.0,
							bidirectional=False)
		self.layer_norm2 = nn.LayerNorm(hidden_size)
		self.dropout2 = nn.Dropout(dropout)
		self.final_fc = nn.Linear(hidden_size, num_classes)
		self.gelu = nn.GELU()


	def __init_hidden(self, batch_size):
		n, hs = self.num_layers, self.hidden_size
		return (torch.zeros(n*1, batch_size, hs),
				torch.zeros(n*1, batch_size, hs))

	def forward(self, t, hidden):
		t = t.squeeze(1)
		# print(t.shape)		
		# t = pack_padded_sequence(t)
		t = self.cnn(t)
		t = self.dense(t)
		t = t.transpose(0,1)
		out, (hn, cn)= self.lstm(t, hidden)
		t = self.dropout2(self.gelu(self.layer_norm2(out)))
		# t = t.transpose(0,1)
		# t = t.transpose(1,2)
		t = self.final_fc(t)

		return t, (hn, cn)

	def step(self, batch):
		t, y, t_lengths, y_lengths = batch[0], batch[1], batch[2], batch[3]
		batch_size = t.shape[0]
		hidden = self.__init_hidden(batch_size)
		hn, c0 = hidden[0].to(device), hidden[1].to(device)
		t, _ = self(t, (hn, c0))
		t = F.log_softmax(t, dim=2).to(torch.float64)
		loss = self.criterion(t, y, t_lengths, y_lengths)
		return loss


	def training_step(self, batch, batch_idx):
		loss = self.step(batch)
		logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr'] }
		self.log('Training Loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return {'loss': loss, 'log': logs}

	# def training_step(self, batch, batch_idx):
	# 	loss = self.step(batch)
	# 	logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr'] }
	# 	return {'loss': loss, 'log': logs}

	def validation_step(self, batch, batch_idx):
		loss = self.step(batch)
		logs = {'val_loss': loss}
		self.current_val_loss = int(loss.item()*100)/100
		self.log('Validation Loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return {'val_loss': loss, 'logs': logs}

	# def validation_step(self, batch, batch_idx):
	# 	loss = self.step(batch)
	# 	return {'val_loss': loss}

	def validation_epoch_end(self, outputs):
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		self.scheduler.step(avg_loss)
		tensorboard_logs = {'val_loss': avg_loss}
		return {'val_loss': avg_loss, 'log': tensorboard_logs}

	def configure_optimizers(self):
		self.optimizer =  optim.Adam(self.parameters(), lr=self.lr)
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                        self.optimizer, mode='min',
                                        factor=0.50, patience=6)

		return {'optimizer': self.optimizer, 'scheduler': self.scheduler}

	def train_dataloader(self):
		return train_dataloader

	def val_dataloader(self):
		return val_dataloader

	def test_dataloader(self):
		return test_dataloader

	def checkpoint_callback(self, checkpoint_path):
		return ModelCheckpoint(
			save_top_k=2,
			auto_insert_metric_name=True,
			verbose=True,
			monitor='Validation Loss',
			mode='min',
			filename=f'kazakh-speech-{self.current_epoch}-{self.current_val_loss}',
			dirpath=checkpoint_path,
		)

	def get_checkpoint_file(self, checkpoint_path):
		checkpoint_file = glob(checkpoint_path + '*.ckpt')
		if checkpoint_file == []:

			return None
		return checkpoint_file[-1]


def train():
	h_params = SpeecRecognition.hyper_parameters
	model = SpeecRecognition(**h_params).to(device)
	logger = TensorBoardLogger('logs', name='speech_recognition')
	trainer = Trainer(logger=logger)	

	trainer = Trainer(
		callbacks=[model.checkpoint_callback(checkpoint_path)],
		max_epochs=epochs, gpus=1,
		logger=logger, gradient_clip_val=1.0, 
		checkpoint_callback=True,
		resume_from_checkpoint=model.get_checkpoint_file(checkpoint_path),
		auto_select_gpus=True,
		num_nodes=1
	)

	trainer.fit(model)





checkpoint_path = 'saved_models/'
epochs = 200


train()

