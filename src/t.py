import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------
# 1️⃣ 載入資料集成 DataFrame
# ------------------------------
def load_dataset(data_dir):
    """
    讀取資料夾中的影像檔路徑與標籤，回傳 DataFrame
    data_dir: 資料夾路徑 (相對於專案根目錄)
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", data_dir)
    filepaths = []
    labels = []

    for label in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for img in os.listdir(class_dir):
            filepaths.append(os.path.join(class_dir, img))
            labels.append(label)

    df = pd.DataFrame({
        "filepath": filepaths,
        "label": labels
    })

    return df

# ------------------------------
# 2️⃣ 建立資料生成器
# ------------------------------
def get_data_generators(train_df, val_df, test_data_dir, target_size=(150,150), batch_size=32):
    """
    建立訓練、驗證、測試生成器
    """
    # 訓練資料生成器，含資料增強
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # 驗證與測試資料生成器，不做增強
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True
    )

    val_generator = test_datagen.flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    test_dir_full = os.path.join(os.path.dirname(__file__), "..", test_data_dir)
    test_generator = test_datagen.flow_from_directory(
        test_dir_full,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_generator, val_generator, test_generator

# ------------------------------
# 3️⃣ 測試程式
# ------------------------------
if __name__ == "__main__":
    # 讀資料
    df = load_dataset("dataset/training_set")
    print("Total samples:", len(df))
    print("Unique labels:", df['label'].unique())

    # 切訓練集、驗證集
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )

    print("Train samples:", len(train_df))
    print("Val samples:", len(val_df))

    # 建立生成器
    train_gen, val_gen, test_gen = get_data_generators(
        train_df, val_df, test_data_dir="dataset/test_set", target_size=(150,150), batch_size=4
    )

    # 取一個 batch 測試
    x_batch, y_batch = next(train_gen)
    print("Image batch shape:", x_batch.shape)   # (4, 150, 150, 3)
    print("Label batch shape:", y_batch.shape)   # (4,)
    print("Label values:", y_batch)