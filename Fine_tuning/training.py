import os
import json
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
import lightning.pytorch as ptl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


# =========================
# CONFIG
# =========================

LANGUAGE = "IT"
DATASET_DIR = "datasets"
CHECKPOINT_DIR = "checkpoints"

train_manifest = os.path.join(DATASET_DIR, "train_manifest.jsonl")
validation_manifest = os.path.join(DATASET_DIR, "validation_manifest.jsonl")
test_manifest = os.path.join(DATASET_DIR, "test_manifest.jsonl")

BASE_MODEL_PATH = os.path.join("models", "stt_en_quartznet15x5.nemo")
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last.ckpt")

EPOCHS = 50
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 16


# =========================
# UTILS
# =========================

def get_charset(manifest_path):
    charset = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Computing character set"):
            row = json.loads(line)
            for ch in row.get("text", ""):
                charset.add(ch)

    charset = sorted(list(charset))
    if "_" in charset:
        charset.remove("_")
    charset.append("_")
    return charset


def enable_bn_se(m):
    if isinstance(m, nn.BatchNorm1d) or "SqueezeExcite" in type(m).__name__:
        m.train()
        for p in m.parameters():
            p.requires_grad_(True)


class UnfreezeEncoderCallback(Callback):
    def __init__(self, unfreeze_epoch=10):
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            print(f"\n[Callback] Unfreezing encoder at epoch {trainer.current_epoch}\n")
            pl_module.encoder.unfreeze()
            pl_module.encoder.train()


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    # -------- 1. VOCAB --------
    print("\n--- Setting up vocabulary ---")
    labels = get_charset(train_manifest)
    print(f"Vocabulary size: {len(labels)}")

    # -------- 2. LOAD MODEL --------
    print("\n--- Loading base model ---")
    assert os.path.exists(BASE_MODEL_PATH), "Base model not found"

    model = nemo_asr.models.ASRModel.restore_from(
        BASE_MODEL_PATH,
        map_location="cpu"
    )

    model.change_vocabulary(new_vocabulary=labels)
    model.cfg.labels = labels

    # -------- 3. FREEZE ENCODER --------
    print("\n--- Freezing encoder ---")
    model.encoder.freeze()
    model.encoder.apply(enable_bn_se)

    cfg = copy.deepcopy(model.cfg)

    # -------- 4. DATASETS --------
    print("\n--- Setting up datasets ---")
    with open_dict(cfg):

        cfg.train_ds.manifest_filepath = train_manifest
        cfg.train_ds.labels = labels
        cfg.train_ds.batch_size = BATCH_SIZE_TRAIN
        cfg.train_ds.num_workers = 4
        cfg.train_ds.pin_memory = True
        cfg.train_ds.trim_silence = True
        cfg.train_ds.normalize_transcripts = False

        cfg.validation_ds.manifest_filepath = validation_manifest
        cfg.validation_ds.labels = labels
        cfg.validation_ds.batch_size = BATCH_SIZE_VAL
        cfg.validation_ds.num_workers = 4
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.trim_silence = True
        cfg.validation_ds.normalize_transcripts = False

        cfg.test_ds = copy.deepcopy(cfg.validation_ds)
        cfg.test_ds.manifest_filepath = test_manifest

    model.cfg = cfg
    model.setup_training_data(cfg.train_ds)
    model.setup_validation_data(cfg.validation_ds)
    model.setup_test_data(cfg.test_ds)

    # -------- 5. OPTIMIZER --------
    print("\n--- Optimizer ---")
    with open_dict(cfg.optim):
        cfg.optim.lr = 5e-4
        cfg.optim.betas = [0.95, 0.5]
        cfg.optim.weight_decay = 1e-3
        cfg.optim.sched.warmup_ratio = 0.05
        cfg.optim.sched.min_lr = 1e-5

    # -------- 6. METRICS --------
    model.wer.use_cer = True
    model.wer.log_prediction = False

    # -------- 7. CALLBACKS --------
    unfreeze_cb = UnfreezeEncoderCallback(unfreeze_epoch=10)

    ckpt_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="asr-epoch={epoch:02d}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # -------- 8. TRAINER --------
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = ptl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=EPOCHS,
        precision="16-mixed" if accelerator == "gpu" else "32",
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        callbacks=[unfreeze_cb, ckpt_cb],
    )

    model.set_trainer(trainer)

    # =========================
    # >>>>>> RESUME QUI <<<<<<
    # =========================

    ckpt_path = LAST_CKPT_PATH if os.path.exists(LAST_CKPT_PATH) else None
    print(f"\n--- Resuming from checkpoint: {ckpt_path} ---\n")

    trainer.fit(model, ckpt_path=ckpt_path)

    # -------- 9. TEST CER --------
    print("\n--- Final CER ---")
    trainer.test(model)

    # -------- 10. TEST WER --------
    model.wer.use_cer = False
    print("\n--- Final WER ---")
    trainer.test(model)

    # -------- 11. SAVE FINAL MODEL --------
    save_path = f"Model-{LANGUAGE}-finetuned.nemo"
    model.save_to(save_path)
    print(f"\nModel saved to: {save_path}")
