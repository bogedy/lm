import optuna
import subprocess
import json
import csv
from pathlib import Path

def objective(trial):
    model_dir = f"trial2_{trial.number}"
    params = {
        "n_layers": 2,
        "n_heads": 2,
        "embed_size": trial.suggest_int("embed_size", 32, 128, step=32),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "seq_len": trial.suggest_categorical("seq_len", [64, 128, 256]),
        "gradient_clipping": trial.suggest_float("gradient_clipping", 0.1, 5.0),
        "init_method": trial.suggest_categorical("init_method", ["normal", "xavier", "kaiming"]),
        "with_residuals": trial.suggest_categorical("with_residuals", [True, False]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.2),
        "model_dir": model_dir
    }

    # Run training as subprocess
    cmd = [
        "python", "main.py",
        "--n_layers", str(params["n_layers"]),
        "--n_heads", str(params["n_heads"]),
        "--embed_size", str(params["embed_size"]),
        "--learning_rate", str(params["learning_rate"]),
        "--batch_size", str(params["batch_size"]),
        "--num_batches_to_train", "5000",
        "--model_dir", params['model_dir'],
        "--seq_len", str(params["seq_len"]),
        "--gradient_clipping", str(params["gradient_clipping"]),
        "--init_method", params["init_method"],
        "--dropout_rate", str(params["dropout_rate"]),
        "--num_batches_to_train", "5000",
        "--model_dir", params['model_dir']
    ]

    if params["with_residuals"]:
        cmd.append("--with_residuals")

    subprocess.run(cmd, check=True)

    # Find latest output dir and read results
    output_dir = Path("checkpoints") / model_dir
    with open(output_dir/"losses.csv") as f:
        last_line = list(csv.reader(f))[-1]
        test_loss = float(last_line[2])

    return test_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, n_jobs = 20)

    print("Best trial:")
    trial = study.best_trial
    print(f"Test loss: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Save importance analysis
    importances = optuna.importance.get_param_importances(study)
    with open("hyperparam_importance2.txt", "w") as f:
        for param, importance in importances.items():
            f.write(f"{param}: {importance}\n")