import os
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_auc_score

from model_EfficientNetB0 import create_model
from split import load_dataset, get_data_generators

gpu = tf.config.list_physical_devices('GPU')
if gpu:
    for g in gpu:
        tf.config.experimental.set_memory_growth(g, True) # 控制 GPU 按需分配

def calculate_metrics(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    auc = roc_auc_score(y_true, y_pred)

    return accuracy, sensitivity, specificity, precision, recall, f1, auc

def train_kfold():
    train_dir = "dataset/training_set"
    df = load_dataset(train_dir)
    X = df.index.values                # 取得 DF 的所有資料編號 array

    kflod = KFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kflod.split(X)):
        print(f'\n===== Fold {fold+1} =====')

        train_df = df.iloc[train_idx]   # 取得 train_idx 的橫列，並回傳一個 DF
        val_df = df.iloc[val_idx]

        train_gen, val_gen = get_data_generators(train_df, val_df)

        model = create_model()

        model.fit(train_gen, epochs = 10, validation_data = val_gen, verbose=1)

        y_pred = model.predict(val_gen)
        y_true = val_gen.classes
        metrics = calculate_metrics(y_true, y_pred)

        loss, acc = model.evaluate(val_gen, verbose=0)

        results.append([fold+1, loss, *metrics])

    columns = ["Fold", "Loss", "Accuracy", "Sensitivity", "Specificity", "Precision", "Recall", "F1-score", "AUC"]
    df_results = pd.DataFrame(results, columns=columns)

    avg = df_results.mean()
    avg["Fold"] = "Average"
    df_results = pd.concat([df_results, avg.to_frame().T], ignore_index=True)
    
    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/kflod_results.csv", index=False)

    print("\n===== Final results =====")
    print(df_results)
        
if __name__ == "__main__":
    train_kfold()