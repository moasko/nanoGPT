import os
import subprocess

def prepare_dataset():
    print("Préparation des données...")
    os.system("python data/dataset/prepare_sql_bot.py")

def train_model():
    print("Démarrage de l'entraînement...")
    cmd = [
        "python", "train.py", "config/train_shakespeare_char.py",
        "--out_dir=out-sqlbot",
        "--batch_size=64",
        "--block_size=128",
        "--max_iters=2000",
        "--eval_interval=200",
        "--eval_iters=50",
        "--learning_rate=1e-3",
        "--dropout=0.1",
        "--wandb_log=False",
        "--device=cpu"
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    prepare_dataset()
    train_model()
