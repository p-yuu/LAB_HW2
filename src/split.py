import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(data_dir):
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

def get_data_generators(train_df, val_df, target_size=(150,150), batch_size=32):
    train_datagen = ImageDataGenerator(
        # rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
    )
    test_datagen = ImageDataGenerator(
        # rescale = 1./255
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col = "filepath",
        y_col = "label",
        target_size = target_size,
        batch_size = batch_size,
        class_mode = "binary",
        shuffle = True
    )

    val_generator = test_datagen.flow_from_dataframe(
        val_df,
        x_col = "filepath",
        y_col = "label",
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = False
    )

    return train_generator, val_generator
