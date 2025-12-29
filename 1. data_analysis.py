import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from PIL import Image
from cv2 import imread

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train_new.csv"

# I create a directory ONLY for image figures so that all plots are saved there
FIG_DIR = PROJECT_ROOT / "figures_for_data_analysis"
FIG_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
sns.set(style="whitegrid")


def analysis_1():
    """
    Analysis 1: AdoptionSpeed distribution by Type (Dog vs Cat)
    """

    # I map the numeric Type codes to readable labels for plots
    type_map = {1: "Dog", 2: "Cat"}
    train_df["TypeLabel"] = train_df["Type"].map(type_map)

    # I compute counts of examples for each combination of Type and AdoptionSpeed.
    t1_counts = (
        train_df
        .groupby(["TypeLabel", "AdoptionSpeed"])
        .size()
        .reset_index(name="count")
    )

    # I also compute normalized proportions within each Type to see relative distribution.
    t1_counts["proportion"] = (
        t1_counts
        .groupby("TypeLabel")["count"]
        .transform(lambda x: x / x.sum())
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=t1_counts,
        x="AdoptionSpeed",
        y="proportion",
        hue="TypeLabel"
    )
    plt.title("T1: AdoptionSpeed distribution by Type")
    plt.xlabel("AdoptionSpeed (0 = same day, 4 = never adopted)")
    plt.ylabel("Proportion within Type")
    plt.legend(title="Type")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T1_adoption_speed_by_type.png", dpi=150)
    plt.close()


def analysis_2():
    """
    Analysis 2: Age vs AdoptionSpeed
    """
    df = train_df.copy()

    # convert months (that i originally have) to years
    df["AgeYears"] = df["Age"] / 12.0

    # I want to clip outliers in YEARS (99th percentile), because they destroy the readibility of the diagram
    upper_yr = np.percentile(df["AgeYears"], 99.9)
    df["AgeYearsClipped"] = df["AgeYears"].clip(upper=upper_yr)

    plt.figure(figsize=(10, 5))

    # dots only (jittered), semi-transparent and small circles
    sns.stripplot(
        data=df,
        x="AdoptionSpeed",
        y="AgeYearsClipped",
        order=[0, 1, 2, 3, 4],
        jitter=0.25,
        size=3,
        alpha=0.35
    )

    # show medians as diamonds so the center is obvious and can be seen
    median = df.groupby("AdoptionSpeed")["AgeYearsClipped"].median()
    plt.plot(
        [0, 1, 2, 3, 4],
        median.loc[[0, 1, 2, 3, 4]].values,
        marker="D",
        linestyle="none",
        markersize=6)

    # taller y-axis than usual. I want to see that there are a lot of animals older than 1,2,3 years
    plt.ylim(0, 8)

    plt.title("T2: Age (years, clipped) by AdoptionSpeed — dots only")
    plt.xlabel("AdoptionSpeed (0 = same day, 4 = never adopted)")
    plt.ylabel("Age (years)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T2_age_years_dots.png", dpi=150)
    plt.close()


def analysis_3():
    """
    Analysis 3: Vaccination status vs AdoptionSpeed
    Very interesting, would never guess this has an anti effect
    """

    # I map the Vaccinated codes to readable labels:
    # 1 = Yes, 2 = No, 3 = Not sure.
    vacc_map = {1: "Yes", 2: "No", 3: "Not sure"}
    train_df["VaccinatedLabel"] = train_df["Vaccinated"].map(vacc_map)

    # I compute counts per (VaccinatedLabel, AdoptionSpeed)
    t3_counts = (
        train_df
        .groupby(["VaccinatedLabel", "AdoptionSpeed"])
        .size()
        .reset_index(name="count")
    )

    # normalize counts
    t3_counts["proportion"] = (
        t3_counts
        .groupby("VaccinatedLabel")["count"]
        .transform(lambda x: x / x.sum())
    )

    # plot vaccination status
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=t3_counts,
        x="VaccinatedLabel",
        y="proportion",
        hue="AdoptionSpeed"
    )
    plt.title("T3: AdoptionSpeed distribution by vaccination status")
    plt.xlabel("Vaccination status")
    plt.ylabel("Proportion within vaccination status")
    plt.legend(title="AdoptionSpeed", loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T3_vaccinated_by_adoption_speed.png", dpi=150)
    plt.close()

    print("[INFO] Tabular EDA plots saved to:", FIG_DIR)


def analysis_4():
    """
    Analysis 4: Here I will analyze how multiple features affect AdoptionSpeed
    features: MaturitySize + FurLength vs AdoptionSpeed
    class proportions for (MaturitySize, FurLength)
    """

    df = train_df.copy()

    maturity_map = {0: "Unknown", 1: "Small", 2: "Medium", 3: "Large", 4: "X-Large"}
    fur_map      = {0: "Unknown", 1: "Short", 2: "Medium", 3: "Long"}

    df["MaturityLabel"] = df["MaturitySize"].map(maturity_map)
    df["FurLabel"] = df["FurLength"].map(fur_map)

    # Combine
    df["Size_Fur"] = df["MaturityLabel"] + " / " + df["FurLabel"]

    # Count for (Size_Fur, AdoptionSpeed)
    group = (
        df.groupby(["Size_Fur", "AdoptionSpeed"])
          .size()
          .rename("count")
          .reset_index()
    )

    # Proportion within each Size_Fur group
    group["group_total"] = group.groupby("Size_Fur")["count"].transform("sum")
    group["proportion"]  = group["count"] / group["group_total"]

    # order categories (looks nicer this way)
    order_speed = [0, 1, 2, 3, 4]
    group["AdoptionSpeed"] = pd.Categorical(group["AdoptionSpeed"], categories=order_speed, ordered=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=group,
        x="Size_Fur",
        y="proportion",
        hue="AdoptionSpeed"
    )
    plt.title("T4: AdoptionSpeed distribution by MaturitySize and FurLength")
    plt.xlabel("Maturity Size / Fur Length")
    plt.ylabel("Proportion within each group")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="AdoptionSpeed", loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T4_size_fur_vs_adoption_speed.png", dpi=150)
    plt.close()


def analysis_5():
    """
    Analysis 5: SentimentScore and SentimentMagnitude — distribution and relation to AdoptionSpeed
    """

    df = train_df.copy()

    # Basic statistics and shape
    print("\n[INFO] Analysis 5 — Sentiment features overview")
    print(f"Shape: {df[['SentimentScore','SentimentMagnitude']].shape}")
    print(df[['SentimentScore','SentimentMagnitude']].describe())

    # Distribution of SentimentScore
    plt.figure(figsize=(7,4))
    sns.histplot(df["SentimentScore"], bins=30, kde=True, color="steelblue")
    plt.title("T5.1: Distribution of SentimentScore")
    plt.xlabel("Sentiment Score (-1 to 1)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T5_1_sentiment_score_dist.png", dpi=150)
    plt.close()

    # Distribution of SentimentMagnitude
    plt.figure(figsize=(7,4))
    sns.histplot(df["SentimentMagnitude"], bins=30, kde=True, color="seagreen")
    plt.title("T5.2: Distribution of SentimentMagnitude")
    plt.xlabel("Sentiment Magnitude (strength of sentiment)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T5_2_sentiment_magnitude_dist.png", dpi=150)
    plt.close()

    # SentimentScore vs AdoptionSpeed
    plt.figure(figsize=(8,5))
    sns.boxplot(
        data=df,
        x="AdoptionSpeed",
        y="SentimentScore",
        order=[0,1,2,3,4],
        palette="Blues"
    )
    plt.title("T5.3: SentimentScore by AdoptionSpeed")
    plt.xlabel("AdoptionSpeed (0 = same day, 4 = never adopted)")
    plt.ylabel("Sentiment Score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T5_3_sentiment_score_vs_adoption.png", dpi=150)
    plt.close()

    # SentimentMagnitude vs AdoptionSpeed
    plt.figure(figsize=(8,5))
    sns.boxplot(
        data=df,
        x="AdoptionSpeed",
        y="SentimentMagnitude",
        order=[0,1,2,3,4],
        palette="Greens"
    )
    plt.title("T5.4: SentimentMagnitude by AdoptionSpeed")
    plt.xlabel("AdoptionSpeed (0 = same day, 4 = never adopted)")
    plt.ylabel("Sentiment Magnitude")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T5_4_sentiment_magnitude_vs_adoption.png", dpi=150)
    plt.close()


    print("[INFO] Sentiment plots saved to:", FIG_DIR)


def analysis_6():
    """
    Analysis 6 — Quantifying how sentiment relates to adoption speed

    I compute Eta-squared (η^2) effect sizes:
         - η^2(SentimentScore ~AdoptionSpeed)
         - η^2(SentimentMagnitude ~AdoptionSpeed)
         η^2 measures how much of the variance in sentiment is explained by adoption speed, lies in [0, 1] range!
         Interpretation: ~0.01 = small, ~0.06 = medium, ~0.14 = large effect

    I do this because η^2 tells me if adoption speed groups are different in their sentiment distributions
    If I have a very small η^2 (< 0.01) -> adoption speed categories explain some small sentiment variance
    """

    df = train_df.copy()
    print("\n[INFO] Analysis 6 — Eta squared")

    def eta_squared(x, group_col):
        grand_mean = x.mean()
        groups = [x[group_col == g] for g in np.unique(group_col)]
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((x - grand_mean) ** 2).sum()
        return ss_between / ss_total if ss_total > 0 else np.nan

    eta_score = eta_squared(df["SentimentScore"], df["AdoptionSpeed"])
    eta_mag = eta_squared(df["SentimentMagnitude"], df["AdoptionSpeed"])
    print(f"η² (SentimentScore ~ AdoptionSpeed): {eta_score:.4f}")
    print(f"η² (SentimentMagnitude ~ AdoptionSpeed): {eta_mag:.4f}")


def analysis_7():
    """
    Analysis 7: STATISTICAL MOMENTS for key numeric features

    In this analysis I focus on 4 statistical moments (mean, variance, skewness, kurtosis) for a few important numeric features:
      - Age in years (derived from months)
      - Fee
      - SentimentScore
      - SentimentMagnitude

    These moments help me understand the
     - central tendency (mean),
     - dispersion (variance),
     - asymmetry (skewness)
     - and tail heaviness (kurtosis)
    of each feature's distribution.
    """

    df = train_df.copy()
    df["AgeYears"] = df["Age"] / 12.0

    features = ["AgeYears", "Fee", "SentimentScore", "SentimentMagnitude"]

    print("\n[INFO] Analysis 7 — Statistical moments")

    for col in features:
        s = df[col].dropna()
        mean_val = s.mean()
        var_val = s.var()
        skew_val = s.skew()
        kurt_val = s.kurtosis()

        print(f"\nFeature: {col}")
        print(f"Mean    : {mean_val:.4f}")
        print(f"Variance: {var_val:.4f}")
        print(f"Skewness: {skew_val:.4f}")
        print(f"Kurtosis: {kurt_val:.4f}")


def analysis_8():
    """
    Analysis 8: Statistical moments per AdoptionSpeed class

    Here I look at how the shape of the distribution changes across target classes for two numeric features:
      - AgeYears
      - SentimentMagnitude

    I compute, for each AdoptionSpeed  (0,1,2,3,4):
      - mean
      - variance
      - skewness
      - kurtosis

    This tells me how certain adoption speed groups of pets have different distributions (example: older pets, heavier tails, more asymmetry maybe)
    """

    df = train_df.copy()
    df["AgeYears"] = df["Age"] / 12.0

    features = ["AgeYears", "SentimentMagnitude"]
    speeds = sorted(df["AdoptionSpeed"].dropna().unique())

    print("\n[INFO] Analysis 8 — Moments per AdoptionSpeed class")

    for col in features:
        print(f"\nFeature: {col}")
        for sp in speeds:
            s = df.loc[df["AdoptionSpeed"] == sp, col].dropna()
            if len(s) == 0:
                continue
            mean_val = s.mean()
            var_val = s.var()
            skew_val = s.skew()
            kurt_val = s.kurtosis()
            print(
                f"AdoptionSpeed = {sp}: "
                f"mean={mean_val:.4f}, var={var_val:.4f}, "
                f"skew={skew_val:.4f}, kurt={kurt_val:.4f}"
            )


def analysis_9():
    """
    Analysis 9: Basic image statistics (resolution, aspect ratio, brightness)
    I use the first image (img1) for each pet and compute:
      - height and width
      - aspect ratio (width / height)
      - mean pixel intensity (as a proxy for brightness)
    """

    df = train_df.copy()
    paths = df["img1"].dropna().values

    heights, widths, ratios, mean_intensities = [], [], [], []

    max_samples = 500  # I only need ~500 images
    print(f"\n[INFO] Analysis 9 — scanning up to {max_samples} images from img1")

    used = 0
    for p in paths:
        if used >= max_samples:
            break
        img = imread(p)
        h, w = img.shape[:2]
        heights.append(h)
        widths.append(w)
        ratios.append(w / h)
        mean_intensities.append(img.mean())
        used += 1

    heights = np.array(heights)
    widths = np.array(widths)
    ratios = np.array(ratios)
    intensities = np.array(mean_intensities)

    print(f"[INFO] Used {used} images")
    print(f"  Height: mean={heights.mean():.1f}, min={heights.min()}, max={heights.max()}")
    print(f"  Width : mean={widths.mean():.1f}, min={widths.min()}, max={widths.max()}")
    print(f"  Aspect ratio (w/h): mean={ratios.mean():.3f}, std={ratios.std():.3f}")
    print(f"  Mean intensity: mean={intensities.mean():.3f}, std={intensities.std():.3f}")


def _load_and_resize_rgb(path, size=(128, 128)):
    """
    Helper to load a image from disk, convert to RGB, resize
    Returns a float32 numpy array of shape (H, W, 3)
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32)
    return arr

def analysis_10():
    """
    Analysis 10: Average image per Type (Dog vs Cat)

    For each Type:
      - 1 = Dog
      - 2 = Cat

    I take 3 images (using paths), resize them, and compute the pixel-wise mean
    This gives an average appearance per class
    """

    df = train_df.copy()

    type_map = {1: "Dog", 2: "Cat"}
    df = df[df["Type"].isin(type_map.keys())].copy()

    size = (128, 128)
    max_per_type = 300  # upper bound per class
    avg_images = {}

    print("\n[INFO] Analysis 10 — Average image per Type (Dog vs Cat)")

    for t, label in type_map.items():
        sub = df[(df["Type"] == t) & df["img1"].notna()]
        imgs = []
        count = 0

        for _, row in sub.iterrows():
            if count >= max_per_type:
                break
            path = row["img1"]
            arr = _load_and_resize_rgb(path, size=size)
            imgs.append(arr)
            count += 1

        stack = np.stack(imgs, axis=0)          # (N, H, W, 3)
        mean_img = stack.mean(axis=0) / 255.0   # scaled to [0,1] for visualization

        avg_images[label] = mean_img

        plt.figure(figsize=(4,4))
        plt.imshow(mean_img)
        plt.axis("off")
        plt.title(f"Average {label} image (n={count})")
        plt.tight_layout()
        out_path = FIG_DIR / f"T10_avg_{label.lower()}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()



if __name__ == "__main__":
    analysis_1()
    # analysis_2()
    # analysis_3()
    # analysis_4()
    # analysis_5()
    # analysis_6()
    # analysis_7()
    # analysis_8()
    # analysis_9()
    analysis_10()










