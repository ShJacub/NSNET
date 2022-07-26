from pytorch_lightning import Trainer
from model.nsnet_model import NSNetModel
from argparse import Namespace
import os

train_dir = './datasets/wav/train'
val_dir = './datasets/wav/val'

hparams = {'train_dir': train_dir,
           'val_dir': val_dir,
           'batch_size': 128,
           'n_fft': 512,
           'n_gru_layers': 3,
           'gru_dropout': 0.2,
           'alpha': 0.35,
           'num_workers' : 48}

model = NSNetModel(hparams=hparams)

trainer = Trainer(gpus=1)
trainer.fit(model)
