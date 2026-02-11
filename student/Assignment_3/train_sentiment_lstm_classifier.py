import os
import random
import sys

import datasets
import gensim
import gensim.downloader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


fasttext = None
emb_dim = None


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def load_fasttext(local_path):
    if os.path.exists(local_path):
        return gensim.models.KeyedVectors.load(local_path, mmap="r")
    model = gensim.downloader.load("fasttext-wiki-news-subwords-300")
    model.save(local_path)
    return model

# Pad or Truncate sentences to exact 32 tokens
def pad_sentence(sentence, max_len=32):
    tokens = sentence.lower().split()
    vecs = []
    for tok in tokens[:max_len]:
        if tok in fasttext:
            vecs.append(fasttext[tok])
        else:
            vecs.append(np.zeros(emb_dim, dtype=np.float32))
    if len(vecs) < max_len:
        pad_len = max_len - len(vecs)
        vecs.extend([np.zeros(emb_dim, dtype=np.float32)] * pad_len)
    return np.stack(vecs, axis=0)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        return self.fc(out)


class LSTMTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def plot_learning_curves(train_loss_history, val_loss_history, train_f1_history, val_f1_history,
                         train_acc_history, val_acc_history, f1_path, acc_path):
    plt.figure(figsize=(12, 15))
    plt.subplot(3, 1, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(train_f1_history, label="Train F1")
    plt.plot(val_f1_history, label="Val F1")
    plt.title("F1 Macro Score Curve")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(val_acc_history, label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f1_path)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(val_acc_history, label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.show()


def plot_confusion_matrix(cm, class_names, path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.show()


def evaluate_on_test(model, test_loader, device, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    test_f1_macro = f1_score(all_labels, all_preds, average="macro")
    test_f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    print("\n" + "=" * 50)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Macro: {test_f1_macro:.4f}")
    print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
    print("=" * 50 + "\n")
    class_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, "outputs/lstm_confusion_matrix.png")


if __name__ == "__main__":
    # set random seeds
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs("outputs", exist_ok=True)
    log_file = open(os.path.join("outputs", "lstm_run.log"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    print("\n========== Loading Dataset ==========")
    # load dataset
    dataset = datasets.load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    print("Dataset loaded.")
    data = pd.DataFrame(dataset["train"])
    data["text_label"] = data["label"].apply(
        lambda x: "positive" if x == 2 else "neutral" if x == 1 else "negative"
    )

    print("\n========== Loading FastText ==========")
    fasttext = load_fasttext("fasttext-wiki-news-subwords-300.kv")
    emb_dim = fasttext.vector_size

    print("\n========== Splitting Data ==========")
    # Train/test split
    X_text = data["sentence"].tolist()
    y = data["label"].tolist()

    X_trainval_text, X_test_text, y_trainval, y_test = train_test_split(
        X_text, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_trainval_text,
        y_trainval,
        test_size=0.15,
        stratify=y_trainval,
        random_state=seed,
    )

    # Pad Sentences
    X_train = np.stack([pad_sentence(s) for s in X_train_text], axis=0)
    X_val = np.stack([pad_sentence(s) for s in X_val_text], axis=0)
    X_test = np.stack([pad_sentence(s) for s in X_test_text], axis=0)

    print("\n========== Building Model ==========")
    num_classes = len(np.unique(y))
    model = LSTMClassifier(input_dim=emb_dim, hidden_dim=128, num_layers=2, num_classes=num_classes)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n========== Preparing DataLoaders ==========")
    train_dataset = LSTMTensorDataset(X_train, y_train)
    val_dataset = LSTMTensorDataset(X_val, y_val)
    test_dataset = LSTMTensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = get_device()
    os.makedirs("outputs", exist_ok=True)
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    print("\n========== Starting Training ==========")
    num_epochs = 30
    best_val_f1 = 0.0
    train_loss_history = []
    val_loss_history = []
    train_f1_history = []
    val_f1_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")
        train_acc = (np.array(all_train_preds) == np.array(all_train_labels)).mean()
        train_loss_history.append(epoch_train_loss)
        train_f1_history.append(train_f1)
        train_acc_history.append(train_acc)

        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_f1 = f1_score(all_val_labels, all_val_preds, average="macro")
        val_acc = (np.array(all_val_preds) == np.array(all_val_labels)).mean()
        val_loss_history.append(epoch_val_loss)
        val_f1_history.append(val_f1)
        val_acc_history.append(val_acc)

        scheduler.step(val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "outputs/best_lstm_model.pth")

    print("\n========== Plotting Learning Curves ==========")
    plot_learning_curves(
        train_loss_history,
        val_loss_history,
        train_f1_history,
        val_f1_history,
        train_acc_history,
        val_acc_history,
        "outputs/lstm_f1_learning_curves.png",
        "outputs/lstm_accuracy_learning_curve.png",
    )

    print("\n========== Evaluating on Test Set ==========")
    evaluate_on_test(model, test_loader, device, "outputs/best_lstm_model.pth")
