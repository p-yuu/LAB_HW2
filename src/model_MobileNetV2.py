from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def create_model(input_shape=(150, 150, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape) # 載入模型
    base_model.trainable = False   # 凍結卷積層
    
    # 建立自定義模型
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 二分類，輸出 0 或 1
    ])
    
    # 編譯模型
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model