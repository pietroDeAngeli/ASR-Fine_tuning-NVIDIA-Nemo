import os
import argparse
import copy

import torch
from omegaconf import open_dict
import lightning.pytorch as ptl
import nemo.collections.asr as nemo_asr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path al modello .nemo")
    parser.add_argument("--test_manifest", type=str, default="datasets/test_manifest.jsonl", help="Manifest di test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Modello non trovato: {args.model}")
    if not os.path.exists(args.test_manifest):
        raise FileNotFoundError(f"Test manifest non trovato: {args.test_manifest}")

    # GPU se disponibile, altrimenti CPU
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    print(f"\n--- Loading model: {args.model} ---")
    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cpu")

    # Config test dataloader (usiamo cfg del modello già salvata nel .nemo)
    cfg = copy.deepcopy(model.cfg)
    with open_dict(cfg):
        # alcuni modelli hanno cfg.test_ds già presente, ma settiamo comunque i campi utili
        if cfg.get("test_ds", None) is None:
            cfg.test_ds = copy.deepcopy(cfg.validation_ds)

        cfg.test_ds.manifest_filepath = args.test_manifest
        cfg.test_ds.batch_size = args.batch_size
        cfg.test_ds.num_workers = args.num_workers
        cfg.test_ds.pin_memory = True
        cfg.test_ds.trim_silence = False

        # IMPORTANT: mantieni coerente con il training (nel tuo training era False)
        cfg.test_ds.normalize_transcripts = False

    model.cfg = cfg
    model.setup_test_data(cfg.test_ds)

    # Logging predizioni (facoltativo)
    model.wer.log_prediction = False

    trainer = ptl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if accelerator == "gpu" else "32",
        logger=False,
        enable_checkpointing=False,
    )

    model.set_trainer(trainer)

    # ---- TEST CER ----
    print("\n--- Running TEST (CER) ---")
    model.wer.use_cer = True
    trainer.test(model=model)

    # ---- TEST WER ----
    print("\n--- Running TEST (WER) ---")
    model.wer.use_cer = False
    trainer.test(model=model)

    print("\nDone.")


if __name__ == "__main__":
    main()
