# [markdown]
# # Finetuning CTC models on other languages
#
# In this tutorial, we fine-tune a pre-trained English ASR model on a new language (Italian) using Mozilla Common Voice data.
# Steps include:
# - Data preprocessing
# - Tokenizer preparation
# - Fine-tuning techniques for low-resource languages
# - Character and subword encoding CTC training


import os
import copy
from omegaconf import OmegaConf, open_dict
import re

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager

import dataset_creation

# --- Audio Preprocessing ---

data_dir = 'datasets/'

clips_dir = os.path.join('datasets', "clips")

# Convert .mp3 to .wav
dataset_creation.convert_mp3_to_wav(clips_dir)

train_file = os.path.join(data_dir, "train.csv")
validation_file = os.path.join(data_dir, "dev.csv")
test_file = os.path.join(data_dir, "test.csv")

# Convert .tsv to .csv with path and sentence columns
train_file, test_file, validation_file = dataset_creation.process_tsv(clips_dir, [train_file, test_file, validation_file])


# Convert to NeMo format
dataset_creation.MVCtoNEMO(train_file, "train")
dataset_creation.MVCtoNEMO(validation_file, "validation")
dataset_creation.MVCtoNEMO(test_file, "test")


data_dir = "datasets"
train_file = os.path.join(data_dir, "train_manifest.json")
validation_file = os.path.join(data_dir, "validation_manifest.json")
test_file = os.path.join(data_dir, "test_manifest.json")

# Normalize text content
dataset_creation.normalize_text(train_file)
dataset_creation.normalize_text(validation_file)
dataset_creation.normalize_text(test_file)


import pandas as pd


LANGUAGE = "it"
tokenizer_dir = os.path.join('tokenizers', LANGUAGE)
manifest_dir = data_dir


train_manifest = f"{manifest_dir}/train/train/train_manifest.json"
dev_manifest = f"{manifest_dir}/validation/validation/validation_manifest.json"
test_manifest = f"{manifest_dir}/test/test/test_manifest.json"


from tqdm.auto import tqdm
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import json

def write_processed_manifest(data, original_path):
    new_manifest_name = os.path.basename(original_path).replace(".json", "_processed.json")
    filepath = os.path.join(os.path.dirname(original_path), new_manifest_name)
    write_manifest(filepath, data)
    print(f"Finished writing manifest: {filepath}")
    return filepath

train_manifest_data = read_manifest(train_manifest)
dev_manifest_data = read_manifest(dev_manifest)
test_manifest_data = read_manifest(test_manifest)

train_text = [data['text'] for data in train_manifest_data]
dev_text = [data['text'] for data in dev_manifest_data]
test_text = [data['text'] for data in test_manifest_data]


#--- Character cleaning---


from collections import defaultdict
import string

def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        for character in row['text']:
            charset[character] += 1
    return charset

def lettere_mancanti(dizionario):
    alfabeto = set(string.ascii_letters)
    lettere_presenti = set(dizionario.keys())
    return sorted(alfabeto - lettere_presenti)

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\…\{\}\(\)\[\]\–\’\'\/]'

def remove_special_characters(data):
    data["text"] = re.sub(chars_to_ignore_regex, '', data["text"]).lower().strip()
    return data

def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])
    print("Finished processing manifest!")
    return manifest

train_charset = get_charset(train_manifest_data)
dev_charset = get_charset(dev_manifest_data)
test_charset = get_charset(test_manifest_data)

print("Missing letters in: train, validation, and test")
print(lettere_mancanti(train_charset))
print(lettere_mancanti(dev_charset))
print(lettere_mancanti(test_charset))
print()

train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))
test_set = set(test_charset.keys())
print(f"Train+Dev charset size: {len(train_dev_set)}")
print(f"Test charset size: {len(test_set)}")
print()

PREPROCESSORS = [remove_special_characters]

print("Reading manifests...")
train_data = read_manifest(train_manifest)
dev_data = read_manifest(dev_manifest)
test_data = read_manifest(test_manifest)
print("Applying filters...")
train_data_processed = apply_preprocessors(train_data, PREPROCESSORS)
dev_data_processed = apply_preprocessors(dev_data, PREPROCESSORS)
test_data_processed = apply_preprocessors(test_data, PREPROCESSORS)

print("Updating manifests...")
train_manifest_cleaned = write_processed_manifest(train_data_processed, train_manifest)
dev_manifest_cleaned = write_processed_manifest(dev_data_processed, dev_manifest)
test_manifest_cleaned = write_processed_manifest(test_data_processed, test_manifest)

print("Recomputing updated charsets...")
train_manifest_data = read_manifest(train_manifest_cleaned)
train_charset = get_charset(train_manifest_data)
dev_manifest_data = read_manifest(dev_manifest_cleaned)
dev_charset = get_charset(dev_manifest_data)
test_manifest_data = read_manifest(test_manifest_cleaned)
test_charset = get_charset(test_manifest_data)

train_dev_set = sorted(set.union(set(train_charset.keys()), set(dev_charset.keys())))
test_set = set(train_charset.keys())

vocaboulary = " ".join(train_dev_set)
print("Updated charset: " + vocaboulary)

#--- Model Training---


char_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_quartznet15x5", map_location='cpu')

freeze_encoder = True

import torch.nn as nn

def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d or 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

if freeze_encoder:
    char_model.encoder.freeze()
    char_model.encoder.apply(enable_bn_se)
    logging.info("Encoder frozen, BatchNorm/SqueezeExcite unfrozen")
else:
    char_model.encoder.unfreeze()
    logging.info("Encoder unfrozen")

char_model.cfg.labels = list(train_dev_set)
cfg = copy.deepcopy(char_model.cfg)


import json
from omegaconf import OmegaConf

cfg_dict = OmegaConf.to_container(cfg, resolve=True)
print(json.dumps(cfg_dict, indent=4))

with open_dict(cfg):
    cfg.train_ds.manifest_filepath = f"{train_manifest_cleaned},{dev_manifest_cleaned}"
    cfg.train_ds.labels = list(train_dev_set)
    cfg.train_ds.normalize_transcripts = False
    cfg.train_ds.batch_size = 8
    cfg.train_ds.num_workers = 0
    cfg.train_ds.pin_memory = True
    cfg.train_ds.trim_silence = True

    cfg.validation_ds.manifest_filepath = test_manifest_cleaned
    cfg.validation_ds.labels = list(train_dev_set)
    cfg.validation_ds.normalize_transcripts = False
    cfg.validation_ds.batch_size = 4
    cfg.validation_ds.num_workers = 0
    cfg.validation_ds.pin_memory = True
    cfg.validation_ds.trim_silence = True

char_model.setup_training_data(cfg.train_ds)
char_model.setup_multiple_validation_data(cfg.validation_ds)


print(OmegaConf.to_yaml(char_model.cfg.optim))

with open_dict(cfg.optim):
    cfg.optim.lr = 0.01
    cfg.optim.betas = [0.95, 0.5]
    cfg.optim.weight_decay = 0.001
    cfg.optim.sched.warmup_steps = None
    cfg.optim.sched.warmup_ratio = 0.05
    cfg.optim.sched.min_lr = 1e-5


print(OmegaConf.to_yaml(char_model.cfg.spec_augment))
char_model.spec_augmentation = char_model.from_config_dict(char_model.cfg.spec_augment)


use_cer = False
log_prediction = True
char_model.wer.use_cer = use_cer
char_model.wer.log_prediction = log_prediction


import torch
import pytorch_lightning as ptl

accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
EPOCHS = 5

trainer = ptl.Trainer(
    devices=1,
    accelerator=accelerator,
    max_epochs=EPOCHS,
    accumulate_grad_batches=1,
    enable_checkpointing=False,
    logger=True,
    log_every_n_steps=5,
    check_val_every_n_epoch=10
)

char_model.set_trainer(trainer)
char_model.cfg = cfg


#%%time
trainer.fit(char_model)


save_path = f"Model-{LANGUAGE}.nemo"
char_model.save_to(f"{save_path}")
print(f"Model saved at: {os.getcwd() + os.path.sep + save_path}")
