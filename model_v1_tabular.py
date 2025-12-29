import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data" / "processed"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001


def load_data():
    """ Loads the pre-split data"""
    train_df = pd.read_csv(DATA_DIR / "train_new.csv")
    val_df = pd.read_csv(DATA_DIR / "validation_new.csv")
    test_df = pd.read_csv(DATA_DIR / "test_new.csv")
    return train_df, val_df, test_df


def preprocess_inputs(train_df, val_df, test_df):
    """
    Important Step: we need Scaled Data.
    We fit scalers ONLY on training data!
    """

    # SEPARATE TARGET
    y_train = train_df.pop('AdoptionSpeed')
    y_val = val_df.pop('AdoptionSpeed')
    y_test = test_df.pop('AdoptionSpeed')

    # FEATURE GROUPS
    # We apply Log Transform to Age, Fee and SentimentMagnitude because they are skewed distributions
    skewed_nums = ['Age', 'Fee', 'SentimentMagnitude']

    # Standard numericals
    other_nums = ['Quantity', 'VideoAmt', 'PhotoAmt', 'SentimentScore']

    # Categoricals (We treat IDs as categories if they are small, or drop them)
    category = ['Type', 'Gender', 'Color1', 'Color2', 'Color3',
            'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
            'Sterilized', 'Health', 'State']

    # Log transform skewed columns manually first
    for df in [train_df, val_df, test_df]:
        for col in skewed_nums:
            # log1p calculates log(1 + x) to handle zeros
            df[col] = np.log1p(df[col])

    # Numerical features: Standardize (Mean 0, Std 1)
    # Categorical features: One Hot Encode
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), skewed_nums + other_nums),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), category)
        ],
        verbose_feature_names_out=False
    )

    # Fit on TRAIN only
    print("[INFO] Fitting preprocessor on Training data...")
    X_train = preprocessor.fit_transform(train_df)

    # FIT - nauci parametre transformacije iz podataka, tj:
    #   - standard scaler nauci μ (sredinu) po koloni
    #   - One hot encoder nauci σ (std) po koloni
    # TRANSFORM: primeni vec naucene parametre na neki konkretan skup (train, test, val) bez ponovnog ucenja.
    # FIT radim samo na train, jer validation i test sluze da mere GENERALIZACIJU. Ako iz njih opet izracunam mi i std ili listu kategorija
    # prenecu "znanje o distribuciji" tih skupova u fazu ucenja, sto nije realno.

    # Transform others
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    print(f"[INFO] Features processed. Input shape: {X_train.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(input_dim):
    """
    Builds a Keras model for tabular data, that I created in 1.data_Creator
    Uses Dropout and BatchNormalization to prevent overfitting on small data
    """

    # Sequential znaci: "imam model koji je prosta linija slojeva, gde svaki sledeci sloj kao INPUT uzima IZLAZ prethodnog"
    # Sequential koristim jer mi treba FEEDFORWARD mreza za tabularne podatke, niz gusto povezanih slojeva.

    # input_dim je broj KOLONA u mom X_train posle preprocesiranja. Posle onog ColumnTransformer dobicu X+train.spahe = (2000, 150) i onda imam 150 kolona = input_dim
    # shape = input_dim znaci da je svaki uzorak (svaki ljubimac) vektor duzine `input_dim`.
    # BATCH DIMENZIJA NIJE NAPISANA, keras ce sam da doda (None, input_dim), gde None znaci bilo koji batch size

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),

        # Ako mi je ulaz vektor X, dimenzije D, a sloj ima 128 neurona, onda:
        #   - W je matrica (D, 128)
        #   - b je vektor  (128, )
        # izlaz:            z = W^T * x + b,
        # pa aktivacija:    a = activation(z)
        tf.keras.layers.Dense(128, activation='relu'),

        # BatchNormalization radi normalizaciju aktivacija.
        # Npr da sam posle aktivacije dobio vrednost "H"
        #   - za svaki neuron u tom sloju, po batch-u se izracuna MEAN i VARIJANSA, da bi dobili h^ = h - μ (batch-a) / σ (batch-a)
        #   - pa se onda normalizuje
        #   - i onda se dodaju neki trainable parametri gamma(Y) i beta(β):   y=Y * h^ + β
        tf.keras.layers.BatchNormalization(),

        # droput je nasumicno gasenje neurona tokom treninga, ideja je da model ne sme da se osloni na 1 neuron, i prinudjen je da raspodeli znanje (smanjuje overfitting)
        tf.keras.layers.Dropout(0.4),

        # Layer 2
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        # Layer 3
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # Output Layer: 5 neurons (Classes 0-4), Softmax for probability
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),

        # ovo je bitno, SPARSE verzija ocekuje da Y bude celobrojni label (0,1,2,3,4), ne ocekuje one-hot vektore, zato nismo radili one-hot kodiranje labela.
        loss='sparse_categorical_crossentropy',  # y is integers (0,1,2,3,4)
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    # 1. Load
    try:
        train_df, val_df, test_df = load_data()
    except FileNotFoundError:
        print("Error: Could not find processed csv files. Run your data script first!")
        exit()

    # 2. Preprocess (scale)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_inputs(train_df, val_df, test_df)

    # 3. Build Model
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    # 4. Callbacks
    callbacks = [
        # Stop training if validation loss doesnt improve for 10 epochs (even this is too much, maybe 5-6 should be better)
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        # Lower learning rate if stuck
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    # 5. Train
    print("\n[INFO] Starting Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Final Evaluation
    print("\n" + "=" * 30)
    print("FINAL TEST EVALUATION")
    print("=" * 30)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 7) Save the model
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    model_out = MODELS_DIR / "model_v1_tabular.keras"
    model.save(model_out)