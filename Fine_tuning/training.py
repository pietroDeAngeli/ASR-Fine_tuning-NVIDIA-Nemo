import os
import json
import copy
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import pytorch_lightning as ptl
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from pytorch_lightning.callbacks import Callback

# --- Configuration ---

LANGUAGE = "IT"
DATASET_DIR = 'datasets'  # Path to the dataset directory

# Manifest paths
train_manifest = os.path.join(DATASET_DIR, "train_manifest.jsonl")
validation_manifest = os.path.join(DATASET_DIR, "validation_manifest.jsonl")
test_manifest = os.path.join(DATASET_DIR, "test_manifest.jsonl")

# Path to the model you want to continue training (or fine-tune)
model_path = os.path.join("models","stt_en_quartznet15x5.nemo") # Make sure this file exists

# --- Helper Functions ---

def get_charset(manifest_path):
    """
    Reads a JSONL manifest file and computes the character set
    (unique characters) from all transcripts.
    Returns a sorted list of characters suitable for NeMo CTC models.
    """
    charset = set()
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Computing character set from {os.path.basename(manifest_path)}"):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue
                
                if 'text' in row:
                    for ch in row['text']:
                        charset.add(ch)
                else:
                    print(f"Skipping line, missing 'text': {line.strip()}")

    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    # Convert to sorted list
    charset = sorted(list(charset))

    # Ensure blank token "_" exists and is LAST
    if "_" in charset:
        charset.remove("_")
    charset.append("_")

    return charset


def enable_bn_se(m):
    """
    Unfreezes BatchNorm and Squeeze&Excite layers for fine-tuning.
    """
    if type(m) == nn.BatchNorm1d or 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

# Callback to unfreeze the encoder
class UnfreezeEncoderCallback(Callback):
    def __init__(self, unfreeze_epoch: int = 5):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.has_unfrozen = False

    def on_train_epoch_start(self, trainer, pl_module):
        # pl_module è il tuo char_model (EncDecCTCModel)
        if (not self.has_unfrozen) and trainer.current_epoch >= self.unfreeze_epoch:
            print(f"\n[Callback] Unfreezing encoder at epoch {trainer.current_epoch}...\n")
            pl_module.encoder.unfreeze()  # rende tutti i parametri trainable
            pl_module.encoder.train()
            self.has_unfrozen = True

# --- Main Training Script ---

if __name__ == "__main__":

    # --- 1. Setup Vocabulary (Labels) ---
    print("--- Setting up Vocabulary ---")
    
    # Build vocabulary ONLY from the training set
    labels = get_charset(train_manifest)
    
    if labels is None:
        raise ValueError(f"Could not read charset from {train_manifest}")

    print(f"Final character set ({len(labels)} chars): {''.join(labels)}")

    # --- 2. Load Pretrained Model ---
    print(f"--- Loading Model From: {model_path} ---")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please check the path.")
        
    char_model = nemo_asr.models.ASRModel.restore_from(model_path, map_location='cpu')

    # --- 3. Setup Fine-Tuning ---
    print("--- Setting up Fine-Tuning ---")
    
    # Freeze the encoder, but unfreeze BN and SE layers for adaptation
    freeze_encoder = True
    if freeze_encoder:
        char_model.encoder.freeze()
        char_model.encoder.apply(enable_bn_se)
        print("Encoder frozen, BatchNorm/SqueezeExcite unfrozen")
    else:
        char_model.encoder.unfreeze()
        print("Encoder unfrozen")

    # Rebuild the decoder for the new vocabulary
    char_model.change_vocabulary(new_vocabulary=labels)

    char_model.cfg.labels = labels

    # Copy the model's config
    cfg = copy.deepcopy(char_model.cfg)

    # --- 4. Setup Dataloaders (Train, Val, Test) ---
    print("--- Setting up Dataloaders ---")
    
    with open_dict(cfg):
        # Setup Train Dataloader
        cfg.train_ds.manifest_filepath = train_manifest
        cfg.train_ds.labels = labels
        cfg.train_ds.normalize_transcripts = False # Already normalized
        cfg.train_ds.batch_size = 8
        cfg.train_ds.num_workers = 4
        cfg.train_ds.pin_memory = True
        cfg.train_ds.trim_silence = True

        # Setup Validation Dataloader
        cfg.validation_ds.manifest_filepath = validation_manifest
        cfg.validation_ds.labels = labels
        cfg.validation_ds.normalize_transcripts = False
        cfg.validation_ds.batch_size = 4
        cfg.validation_ds.num_workers = 4
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.trim_silence = True
        
        # Best Practice: Setup Test Dataloader (for final evaluation)
        cfg.test_ds = copy.deepcopy(char_model.cfg.validation_ds)
        cfg.test_ds.manifest_filepath = test_manifest
        cfg.test_ds.labels = labels
        cfg.test_ds.normalize_transcripts = False
        cfg.test_ds.batch_size = 4
        cfg.test_ds.num_workers = 4

    char_model.cfg = cfg

    char_model.setup_training_data(cfg.train_ds)
    char_model.setup_multiple_validation_data(cfg.validation_ds)
    char_model.setup_multiple_test_data(cfg.test_ds)

    # --- 5. Setup Optimizer and Augmentation ---
    print("--- Setting up Optimizer ---")

    with open_dict(cfg.optim):
        cfg.optim.lr = 0.01
        cfg.optim.betas = [0.95, 0.5]
        cfg.optim.weight_decay = 0.001
        cfg.optim.sched.warmup_steps = None
        cfg.optim.sched.warmup_ratio = 0.05
        cfg.optim.sched.min_lr = 1e-5

    print(OmegaConf.to_yaml(char_model.cfg.optim))
    
    if cfg.spec_augment:
        print("--- Setting up SpecAugment ---")
        char_model.spec_augmentation = char_model.from_config_dict(cfg.spec_augment)

    # Set the metric (object 'wer') to use CER
    print("--- Setting up Metrics ---")
    char_model.wer.use_cer = True
    char_model.wer.log_prediction = True
    print("Metric set to CER for training and initial testing.")

    # --- 6. Setup Trainer ---
    print("--- Setting up Trainer ---")
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 50
    
    UNFREEZE_EPOCH = 1 # Change to 5
    unfreeze_cb = UnfreezeEncoderCallback(unfreeze_epoch=UNFREEZE_EPOCH)
    
    trainer = ptl.Trainer(
        devices=1,
        accelerator=accelerator,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        enable_checkpointing=True,
        logger=True,
        log_every_n_steps=5,
        check_val_every_n_epoch=5,
        callbacks=[unfreeze_cb],
    )
    
    char_model.set_trainer(trainer)
    char_model.cfg = cfg

    # --- 7. Start Training ---
    print(f"--- Starting Training ({EPOCHS} Epochs) ---")
    trainer.fit(char_model)

    # --- 8. Final Test (CER) ---
    print("--- Training Finished. Running Final Test (CER) ---")
    if hasattr(char_model, 'test_dataloaders') and cfg.test_ds.manifest_filepath:
        trainer.test(model=char_model, dataloaders=char_model.test_dataloaders())
    else:
        print("Skipping CER test phase (test manifest not configured).")

    # --- 9. ADDED: Final Test (WER) ---
    print("--- Re-configuring for WER Test ---")
    
    # Re-configure the model's metric to calculate WER
    char_model.wer.use_cer = False
    print("Metric set to WER.")

    if hasattr(char_model, 'test_dataloaders') and cfg.test_ds.manifest_filepath:
        # Re-run the test (the trainer will now use the new metric configuration)
        trainer.test(model=char_model, dataloaders=char_model.test_dataloaders())
    else:
        print("Skipping WER test phase (test manifest not configured).")


    # --- 10. Save Final Model ---
    print("--- Saving Model ---")
    save_path = f"Model-{LANGUAGE}-finetuned.nemo"
    char_model.save_to(f"{save_path}")
    print(f"Model saved at: {os.getcwd() + os.path.sep + save_path}")
