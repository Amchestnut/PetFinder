# GRAYSCALE, not RGB

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_IMG_DIR = PROJECT_ROOT / "Data" / "train_images"

IMG_SIZE = (128, 128)
K_IMAGES = 3
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
RANDOM_SEED = 842023
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_data_splits():
    """ Loads the pre-split data"""
    train_df = pd.read_csv(DATA_DIR / "train_new.csv")
    val_df   = pd.read_csv(DATA_DIR / "validation_new.csv")
    test_df  = pd.read_csv(DATA_DIR / "test_new.csv")
    return train_df, val_df, test_df


def generate_image_paths(df):
    """"
    Generates a numpy array of shape (N, 3) containing file paths
    Assumes PetID-1.jpg, PetID-2.jpg, PetID-3.jpg exist
    """
    pet_ids = df['PetID'].values
    paths = []

    for pid in pet_ids:
        # Create 3 paths per pet
        p1 = str(TRAIN_IMG_DIR / f"{pid}-1.jpg")
        p2 = str(TRAIN_IMG_DIR / f"{pid}-2.jpg")
        p3 = str(TRAIN_IMG_DIR / f"{pid}-3.jpg")
        paths.append([p1, p2, p3])

    return np.asarray(paths)


def preprocess_tabular(train_df, val_df, test_df):
    """
    Scales tabular data and prepares targets
    Returns: (X_tab_train, X_paths_train, y_train) and others
    """
    print("[INFO] Preprocessing Tabular Data...")

    # Extract Image Paths BEFORE dropping PetID
    train_paths = generate_image_paths(train_df)
    val_paths   = generate_image_paths(val_df)
    test_paths  = generate_image_paths(test_df)

    # Extract Targets
    y_train = train_df.pop('AdoptionSpeed').astype(int).values
    y_val   = val_df.pop('AdoptionSpeed').astype(int).values
    y_test  = test_df.pop('AdoptionSpeed').astype(int).values

    # Clean Columns (Remove PetID etc)
    cols_to_drop = ['PetID', 'img1', 'img2', 'img3']
    for df in [train_df, val_df, test_df]:
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    skewed_nums = ['Age', 'Fee', 'SentimentMagnitude']

    other_nums = ['Quantity', 'VideoAmt', 'PhotoAmt', 'SentimentScore']

    categories = ['Type', 'Gender', 'Color1', 'Color2', 'Color3',
                'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                'Sterilized', 'Health', 'State']

    for df in (train_df, val_df, test_df):
        for col in skewed_nums:
            df[col] = np.log1p(df[col])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), skewed_nums + other_nums),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categories)
        ],
        verbose_feature_names_out=False
    )

    # Fit on Train, transform others
    X_train_tab = preprocessor.fit_transform(train_df)
    X_val_tab   = preprocessor.transform(val_df)
    X_test_tab  = preprocessor.transform(test_df)

    print(f"[INFO] Tabular Input Shape: {X_train_tab.shape}")

    return (X_train_tab, train_paths, y_train), \
           (X_val_tab,   val_paths,   y_val), \
           (X_test_tab,  test_paths,  y_test)


def create_dataset(X_tab, X_img_paths, y, is_train=True):
    """
    Returns image_input of shape (K=3, H, W, 1) per example (GRAYSCALE)
    """

    def load_one_image(path):
        # decode as GRAYSCALE (1 channel)
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, IMG_SIZE)
        return img

    def load_three_images(paths_3):
        # paths_3: (3,) tf.string
        imgs = tf.map_fn(load_one_image, paths_3, fn_output_signature=tf.float32)
        # imgs shape: (3, H, W, 1)
        return imgs

    def map_func(tab_data, paths_row, target):
        imgs = load_three_images(paths_row)
        return {"tabular_input": tab_data, "image_input": imgs}, target

    ds = tf.data.Dataset.from_tensor_slices((X_tab, X_img_paths, y))

    if is_train:
        ds = ds.shuffle(2048)

    ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_custom_multimodal_model(tabular_dim):
    """
    TimeDistributed CNN over K grayscale images
    Input: image_input (None, K, 128, 128, 1)
    """
    image_input = tf.keras.Input(shape=(K_IMAGES, IMG_SIZE[0], IMG_SIZE[1], 1), name="image_input")
    tabular_input = tf.keras.Input(shape=(tabular_dim,), name="tabular_input")

    # PART 1: Custom CNN
    x = image_input
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Rescaling(1./255))(x)

    # Conv blocks (per image)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    # Aggregate across K images
    image_features = tf.keras.layers.GlobalAveragePooling1D()(x)

    # PART 2: Tabular
    t = tf.keras.layers.Dense(128, 'relu')(tabular_input)
    t = tf.keras.layers.BatchNormalization()(t)
    t = tf.keras.layers.Dropout(0.3)(t)
    t = tf.keras.layers.Dense(64, 'relu')(t)
    t = tf.keras.layers.BatchNormalization()(t)

    # PART 3: Merge
    combined = tf.keras.layers.concatenate([t, image_features])

    # head
    z = tf.keras.layers.Dense(64, activation='relu')(combined)
    z = tf.keras.layers.Dropout(0.3)(z)
    out = tf.keras.layers.Dense(5, activation='softmax')(z)

    model = tf.keras.Model(inputs=[tabular_input, image_input], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    # 1) Load data
    train_df, val_df, test_df = load_data_splits()

    # 2) Preprocess
    (X_train_tab, X_train_paths, y_train), \
    (X_val_tab,   X_val_paths,   y_val), \
    (X_test_tab,  X_test_paths,  y_test) = preprocess_tabular(train_df, val_df, test_df)

    # 3. Create TensorFlow Datasets
    print("[INFO] Creating TF Datasets...")
    train_ds = create_dataset(X_train_tab, X_train_paths, y_train, is_train=True)
    val_ds   = create_dataset(X_val_tab,   X_val_paths,   y_val,   is_train=False)
    test_ds  = create_dataset(X_test_tab,  X_test_paths,  y_test,  is_train=False)

    # 4) Build model
    model = build_custom_multimodal_model(tabular_dim=X_train_tab.shape[1])
    model.summary()

    print("\n[INFO] Starting Custom Multimodal Training...")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    # 5) Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Evaluate
    print("\n" + "=" * 30)
    print("FINAL TEST EVALUATION")
    print("=" * 30)
    loss, acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # 7. Save
    model.save(PROJECT_ROOT / "models" / "model_v4_multimodal.keras")
    print("[INFO] Model saved.")
