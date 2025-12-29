from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 842023  # RN 84/23
np.random.seed(RANDOM_SEED)

# Paths are relative to project root (where this script is located)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"

TRAIN_CSV_PATH = DATA_DIR / "train.csv"
TRAIN_SENTIMENT_CSV_PATH = DATA_DIR / "train_sentiment.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_images"

# Target size of the subset of rows that I will use in this project. This is really a concern, because only 400 entries for class 0 !! 3000 is too unbalanced, 2000 is too few
TARGET_SUBSET_SIZE = 2500


def load_train():
    """
    Load train.csv file
    """

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"[INFO] Train shape: {train_df.shape}")

    return train_df


def basic_overview(df: pd.DataFrame):
    """
    Print basic info about the dataframe that I loaded
    """

    print("\n" + "=" * 60)
    print(f"[BASIC OVERVIEW]")
    print("=" * 60)

    print("\n[HEAD]")
    print(df.head())

    # I want to see all columns that we have in this dataframe (new df is reduced).
    print("\n[COLUMNS]")
    print(df.columns.tolist())

    print("\n[INFO]")
    df.info()

    if "AdoptionSpeed" in df.columns:
        print("\n[AdoptionSpeed value distribution, from 0 to 4]")
        print(df["AdoptionSpeed"].value_counts().sort_index())

        print("\n[AdoptionSpeed value counts (normalized, in percents %), sum up to 1]")
        print(df["AdoptionSpeed"].value_counts(normalize=True).sort_index())

    # NA ratio of each column, very important, I will later drop each that has NA!
    print("\n[====== NA ratio per column ======]")
    na_ratio = df.isna().sum()
    print(na_ratio)


def load_and_merge_sentiment(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load train_sentiment.csv dataset and merge it with train using PetID
    I will just append (SentimentScore, SentimentMagnitude) to the result dataset, using left join with PetID
    """

    sent_df = pd.read_csv(TRAIN_SENTIMENT_CSV_PATH)
    print(f"[INFO] Sentiment shape: {sent_df.shape}")

    print("[INFO] Sentiment columns:")
    print(sent_df.columns.tolist())

    print("\n[Sentiment HEAD]")
    print(sent_df.head())

    # columns for sentiment, everything beside PetID col
    cols = [c for c in sent_df.columns if c != "PetID"]
    if not cols:
        return train_df

    # left join using PetID
    merged = train_df.merge(sent_df, on="PetID", how="left")
    return merged


def create_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create a subset of rows (3000) suitable for training my 3 models later, prerequisites:
    - has at least 3 photos (PhotoAmt >= 3)
    - has non-NaN sentiment columns

    Also, we can drop some things like Name, RescuerId and description. Maybe something else as well, but will see later
    """
    print("\n" + "=" * 60)
    print("[CREATING SUBSET...]")
    print("=" * 60)

    # Make a copy so I dont touch the original
    out = df.copy()

    # Drop columns that are not needed
    out = out.drop(columns=["Name", "Description", "RescuerID"], errors="ignore")

    # Keep only rows with at least 3 photos, with a simple filter
    if "PhotoAmt" in out.columns:
        out = out[out["PhotoAmt"] >= 3]

    # I dont want NaN sentiments, so if some row has NaN, exclude it
    sentiment_cols = [c for c in out.columns if "Sentiment" in c]
    for col in sentiment_cols:      # 2 kolone, prvo uzima da je COLUMN = SENTIMENTSCORE,   pa uzima COLUMN = SENTIMENT_MAGNITUDE
        out = out[out[col].notna()]

    def get_paths(pet_id):
        # Create paths for image 1, 2, and 3
        p1 = TRAIN_IMG_DIR / f"{pet_id}-1.jpg"
        p2 = TRAIN_IMG_DIR / f"{pet_id}-2.jpg"
        p3 = TRAIN_IMG_DIR / f"{pet_id}-3.jpg"
        return pd.Series([str(p1), str(p2), str(p3)])

    print("[INFO] Checking file existence for 3 images per pet... (this may take a moment)")
    # Create the 3 new columns
    out[["img1", "img2", "img3"]] = out["PetID"].apply(get_paths)

    # Drop rows where we couldn't find all 3 images, and drop unneeded columns
    out = out.dropna(subset=["img1", "img2", "img3"])
    out = out.drop(columns=["Name", "Description", "RescuerID"], errors="ignore")

    print(f"[INFO] Rows with 3 valid images found: {len(out)}")
    return out


def split_train_val_test_simple(df: pd.DataFrame):
    """
    I wont use this, but just highlights the main idea
    Split df into train, test and validation dataset without stratification, not really smart, but can pass
    """

    # take 3000 random rows
    df_sample = df.sample(n=TARGET_SUBSET_SIZE, random_state=RANDOM_SEED)

    # split 80% / 10% / 10%
    train_df, temp_df = train_test_split(df_sample, test_size=0.2, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)

    print(f"[INFO] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def make_fixed_sample(df: pd.DataFrame,
                      total_size: int = TARGET_SUBSET_SIZE) -> pd.DataFrame:
    """
    First we take all rows from class 0, and then we take randomly from classes 1-4, but keep their class ratios
    So ideally, I want exactly (total_size - n0) rows from df_rest, stratified by target (2500 - 300 = need 2200)
    """
    df0 = df[df["AdoptionSpeed"] == 0]
    n0 = len(df0)

    df_rest = df[df["AdoptionSpeed"] != 0]

    # So ideally, I want exactly (total_size - n0) rows from df_rest, stratified by target (2000 - 200 = need 1800)
    need = total_size - n0

    # train_test_split to pick a stratified sample of size need
    # test_size is the fraction we want to TAKE from df_rest
    take_frac = need / len(df_rest)
    _, sample_rest = train_test_split(
        df_rest,
        test_size=take_frac,
        random_state=RANDOM_SEED,
        stratify=df_rest["AdoptionSpeed"]
    )

    # combine: all class 0 rows + the sampled rest
    df_sample = pd.concat([df0, sample_rest], ignore_index=True)

    # shuffle, it would be smart here
    df_sample = df_sample.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return df_sample


def split_train_val_test(df: pd.DataFrame,
                         val_size: float = 0.1,
                         test_size: float = 0.1):
    """
    Take 3000 random rows (stratified if possible)
    Split df into train/val/test dataset, with stratification on AdoptionSpeed (80 / 10 / 10)
    Better than just simple split
    """

    print("\n" + "=" * 60)
    print("[SPLIT TRAIN / VAL / TEST]")
    print("=" * 60)

    df_sample = make_fixed_sample(df, total_size=TARGET_SUBSET_SIZE)

    # create y (which is the target)
    y = df_sample["AdoptionSpeed"]

    # I split 80% train, 20% temp (val + test)
    df_train_val, df_test = train_test_split(
        df_sample,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y  # keeps same class ratios
    )

    y_train_val = df_train_val["AdoptionSpeed"]
    val_frac = val_size / (1 - test_size)

    # u sustini uzmem ~11.11% redova od df_train_val da bude DF_VALIDATION, a ostatak je DF_TRAIN
    df_train, df_validation = train_test_split(
        df_train_val,
        test_size=val_frac,
        random_state=RANDOM_SEED,
        stratify=y_train_val
    )

    print(f"[INFO] Train: {len(df_train)}, Validation: {len(df_validation)}, Test: {len(df_test)}")
    return df_train, df_validation, df_test



if __name__ == "__main__":
    # 1) Load base train.csv
    train_df = load_train()

    # 2) Basic overview BEFORE sentiment merge
    basic_overview(train_df)

    # 3) Merge sentiment
    train_merged = load_and_merge_sentiment(train_df)

    # 4) Basic overview AFTER sentiment merge
    print("\n AFTER SENTIMENT OVERVIEW:")
    basic_overview(train_merged)

    # 5) Create subset data
    subset_df = create_subset(train_merged)

    # 6) basic overview of the subset data
    print("\n SUBSET OVERVIEW:")
    basic_overview(subset_df)

    print("\n[INFO] subset_df head():")
    print(subset_df.head())

    # 7) Split subset into train/val/test dataset
    train_mm, val_mm, test_mm = split_train_val_test(subset_df)

    # 8) Just a check to see how much data we have for class 0 at the end (BONUS)
    print("\n TRAIN OVERVIEW:")
    basic_overview(train_mm)

    # 9) Save this splitted data to disk for later
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(exist_ok=True)

    train_mm.to_csv(out_dir / "train_new.csv", index=False)
    val_mm.to_csv(out_dir /   "validation_new.csv", index=False)
    test_mm.to_csv(out_dir /  "test_new.csv", index=False)

    print(f"\n[INFO] Saved processed splits to: {out_dir}")
