from tensorboard.backend.event_processing import event_accumulator
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--logdir",
    type=str,
    default="lightning_logs/version_0",
    help="Path a lightning_logs/version_X"
)
parser.add_argument(
    "--tag",
    type=str,
    default="val_loss",
    help="Tag da estrarre (default: val_loss)"
)
args = parser.parse_args()

assert os.path.exists(args.logdir), f"Directory non trovata: {args.logdir}"

ea = event_accumulator.EventAccumulator(args.logdir)
ea.Reload()

assert args.tag in ea.Tags()["scalars"], f"Tag {args.tag} non trovato"

print("global_step val_loss")
for ev in ea.Scalars(args.tag):
    print(ev.step, ev.value)
