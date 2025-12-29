import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path

from sklearn.utils import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_IMG_DIR = PROJECT_ROOT / "Data" / "train_images"
RANDOM_SEED = 842023

# I use 128x128 for resize, because this is training from 0, bigger numXnum would be too slow
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001


def load_data_splits():
    """ Loads the pre-split data"""
    train_df = pd.read_csv(DATA_DIR / "train_new.csv")
    val_df = pd.read_csv(DATA_DIR / "validation_new.csv")
    test_df = pd.read_csv(DATA_DIR / "test_new.csv")
    return train_df, val_df, test_df


def generate_image_paths(df):
    """
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

    return np.array(paths)

def preprocess_tabular(train_df, val_df, test_df):
    """
    Scales tabular data and prepares targets
    Returns: (X_tab_train, X_paths_train, y_train) and others
    """
    print("[INFO] Preprocessing Tabular Data...")

    # Extract Image Paths BEFORE dropping PetID
    train_paths = generate_image_paths(train_df)
    val_paths = generate_image_paths(val_df)
    test_paths = generate_image_paths(test_df)

    # Extract Targets
    y_train = train_df.pop('AdoptionSpeed').astype(int).values
    y_val = val_df.pop('AdoptionSpeed').astype(int).values
    y_test = test_df.pop('AdoptionSpeed').astype(int).values

    # Clean Columns (Remove PetID etc)
    cols_to_drop = ['PetID', 'img1', 'img2', 'img3']
    for df in [train_df, val_df, test_df]:
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)


    skewed_nums = ['Age', 'Fee', 'SentimentMagnitude']

    other_nums = ['Quantity', 'VideoAmt', 'PhotoAmt', 'SentimentScore',]

    categories = ['Type', 'Gender', 'Color1', 'Color2', 'Color3',
                'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                'Sterilized', 'Health', 'State']

    for df in [train_df, val_df, test_df]:
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
    X_val_tab = preprocessor.transform(val_df)
    X_test_tab = preprocessor.transform(test_df)

    print(f"[INFO] Tabular Input Shape: {X_train_tab.shape}")

    return (X_train_tab, train_paths, y_train), \
           (X_val_tab, val_paths, y_val), \
           (X_test_tab, test_paths, y_test)


def create_dataset(X_tab, X_img_paths, y, is_train=True):
    """
    Loads 3 images and stacks them into a 9-channel tensor
    """

    def load_and_stack_images(paths):
        images = []
        for i in range(3):
            # Read and decode
            img = tf.io.read_file(paths[i])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            images.append(img)

        # Stack along the last axis (channels)
        # Shape becomes (128, 128, 9)
        combined_img = tf.concat(images, axis=-1)
        return combined_img

    def map_func(tab_data, img_paths_row, target):
        # img_paths_row contains 3 paths
        img_data = load_and_stack_images(img_paths_row)
        return {"tabular_input": tab_data, "image_input": img_data}, target

    dataset = tf.data.Dataset.from_tensor_slices((X_tab, X_img_paths, y))

    if is_train:
        dataset = dataset.shuffle(RANDOM_SEED)

    dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_custom_multimodal_model(tabular_dim):
    # 1. input kaze: "ulazna slika je tenzor (128,128,9)" a
    # 2. input kaze: "tabularni ulaz je vektor duzine tabular_dim"

    # Input shape is now (128, 128, 9) because we have 3 images and on top of that I have x3 channels
    image_input = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 9), name="image_input")
    tabular_input = tf.keras.Input(shape=(tabular_dim,), name="tabular_input")

    # PART 1: Custom CNN
    x = tf.keras.layers.Rescaling(1. / 255)(image_input)                                # skaliranje piksela sa [0,255] na [0,1]

    # We need slightly more filters because the input is more complex (9 channels)
    # kernel je 3x3, relu, a padding je 'same' da ne bi gubio dimenziju pre pooling-a

    # Conv2D uzima male filtere 3x3 i klizi preko slike, i za svaki polozaj filtera uzme prozor 3x3x9,
    # izracuna skalarni proizvod (tezina * piksel + bias) i dobija 1 vrednost (feature) za taj piksel u feature mapi
    # imamo 32 razlicita filtera, a svaki filter uci neki drugi obrazac (jedan hvata horizontalne ivice, drugi vertikalne, treci prelaze svetlo/tamno, itd..)
    # rezultat je shape (128,128,32), ima 32 razlicita filtera
    # 1. sloj ucimo low level signale, ivice, prelaze, kontrast
    # 2. sloj ucimo nad tim iivicama prostije teksture i oblike (sape, mrlje)
    # 3. sloj ucimo delove objekata, npr njuska, cela glava, obris tela, pozadina...
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Sad radimo pooling preko blokova 2x2, i za svaki blok 2x2 uzima MAXIMALNU vrednost, pa cemo posle 2x2 poolinga da imamo shape (64,64,32)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)


    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)         # imacemo Conv2d(64) + MaxPool -> (32,32,64)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)         # imacemo Conv2d(128) + MaxPool -> (16,16,128)


    # GlobalAveragePooling2D uzima feature mac velicine (16,16,128),  za svaki kanal radi "PROSEK PO PROSTORU":
    #   - uzme 1 feature mapu 1 filtera, dimenzije 16x16, sabere svih 16x16 = 256 vrednosti, podelimo sa 256, i dobijemo prost broj -> prosek aktivacije tog filtera
    #   - izlaz je vektor duzine C (128,), samo 128 dimenzija, a ne 16x16x128=32768 vrednosti kao za FLATTEN
    #   - to je KOMPAKTAN EMBEDDING SLIKE, tj lepse receno ovo je "SAZETAK VIZUELNIH INFORMACIJA"
    # MOGAO SAM DA RADIM I FLATTEN, ali ovo smanjuje drasticno broj parametara u sledecem dense sloju i smanjuje sansu za overfitting.
    x_img = tf.keras.layers.GlobalAveragePooling2D()(x)

    # x_img je vektor duzine 128
    # pa onda pravimo jos kompaktniju reprezentaciju, x_img je vektor dimenzije 64
    x_img = tf.keras.layers.Dense(64, activation='relu')(x_img)

    # PART 2: Tabular
    x_tab = tf.keras.layers.Dense(128, activation='relu')(tabular_input)
    x_tab = tf.keras.layers.BatchNormalization()(x_tab)
    x_tab = tf.keras.layers.Dropout(0.3)(x_tab)
    x_tab = tf.keras.layers.Dense(64, activation='relu')(x_tab)

    # PART 3: Merge
    combined = tf.keras.layers.concatenate([x_tab, x_img])

    # Head
    # Jos jedan Dense (64,relu) uci kombinovane obrasce (npr "mlad pas sa svetlom slikom i visokom sentiment magnitudom")
    z = tf.keras.layers.Dense(64, activation='relu')(combined)
    z = tf.keras.layers.Dropout(0.3)(z)
    output = tf.keras.layers.Dense(5, activation='softmax')(z)

    model = tf.keras.Model(inputs=[tabular_input, image_input], outputs=output)
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    # 1. Load data
    train_df, val_df, test_df = load_data_splits()

    # 2. Preprocess
    (X_train_tab, X_train_paths, y_train), \
    (X_val_tab, X_val_paths, y_val), \
    (X_test_tab, X_test_paths, y_test) = preprocess_tabular(train_df, val_df, test_df)

    # added this extra, because the data is a bit unbalanced
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("[INFO] Class weights:", class_weight)

    # 3. Create TensorFlow Datasets
    print("[INFO] Creating TF Datasets...")
    train_ds = create_dataset(X_train_tab, X_train_paths, y_train, is_train=True)
    val_ds = create_dataset(X_val_tab, X_val_paths, y_val, is_train=False)
    test_ds = create_dataset(X_test_tab, X_test_paths, y_test, is_train=False)

    # 4. Build Model
    model = build_custom_multimodal_model(tabular_dim=X_train_tab.shape[1])
    model.summary()

    print("\n[INFO] Starting Custom Multimodal Training...")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)
    ]

    # 5. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # 6. Evaluate
    print("\n" + "=" * 30)
    print("FINAL TEST EVALUATION")
    print("=" * 30)
    loss, acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # 7. Save
    model.save(PROJECT_ROOT / "models" / "model_v2_multimodal.keras")
    print("[INFO] Model saved.")



