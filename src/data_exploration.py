import matplotlib.pyplot as plt
import seaborn as sns


def plot_outliers(df):
    """
    Plots boxplots for numerical columns to detect outliers.
    """
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns

    plt.figure(figsize=(15, 8))
    for i, col in enumerate(num_cols):
        plt.subplot(3, 5, i + 1)  # Arrange subplots in a grid
        sns.boxplot(y=df[col])
        plt.title(col)
        plt.tight_layout()

    plt.show()


def explore_data(df):
    """
    Performs exploratory data analysis.
    """
    print("\nChecking for Outliers...")
    plot_outliers(df)


def check_sex_column(df):
    print("Unique values in 'sex' column:", df["sex"].unique())
