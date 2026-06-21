import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_pandas_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generates an overview of data types, missing records, and unique value counts."""
    summary = pd.DataFrame(
        {
            "Type": df.dtypes,
            "Non-Null Count": df.count(),
            "Missing": df.isna().sum(),
            "% Missing": df.isna().mean() * 100,
            "Unique Values": df.nunique(),
        }
    )
    return summary.sort_values("Unique Values", ascending=False)


def plot_continuous_distributions(df: pd.DataFrame, cols: list[str]) -> None:
    """Plots a 2-row grid of Histograms (with KDE) and Boxplots for continuous columns."""
    if not cols:
        print("Warning: No continuous columns provided for plotting.")
        return

    fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 8), squeeze=False)

    for i, col in enumerate(cols):
        # Histogram + KDE (Row 0)
        sns.histplot(data=df, x=col, kde=True, ax=axes[0, i], bins=30)
        axes[0, i].set_title(f"{col}\nHistogram")

        # Boxplot (Row 1)
        sns.boxplot(data=df, x=col, ax=axes[1, i])
        axes[1, i].set_title(f"{col}\nBoxplot")

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df: pd.DataFrame, cols: list[str], num_columns_grid: int = 3
) -> None:
    """Dynamically creates count plots arranged nicely based on the number of input columns."""
    if not cols:
        print("Warning: No categorical columns provided for plotting.")
        return

    n_cols = len(cols)
    n_rows = (n_cols + num_columns_grid - 1) // num_columns_grid

    fig, axes = plt.subplots(n_rows, num_columns_grid, figsize=(6 * num_columns_grid, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data_col = df[col].astype(str)
        sns.countplot(x=data_col, ax=axes[i], order=data_col.value_counts().index)
        axes[i].set_title(f"{col} - Count Plot")
        axes[i].tick_params(axis="x", rotation=45)

    # Clean up empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_bivariate_relationship(df: pd.DataFrame, feature_col: str, target_col: str):
    """Plots a rate-probability barplot alongside an absolute count crosstab heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Probability Barplot
    sns.barplot(data=df, x=feature_col, y=target_col, ax=axes[0])
    axes[0].set_ylabel(f"Mean Rate of ({target_col})")
    axes[0].set_title(f"{target_col} Rate by {feature_col}")

    # Heatmap Crosstab (Flipped along y-axis)
    ctab = pd.crosstab(df[target_col], df[feature_col])
    sns.heatmap(ctab.iloc[::-1], annot=True, fmt="d", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Volume Distribution Matrix")

    plt.tight_layout()
    plt.show()


def plot_categorical_continuous_matrix(
    df: pd.DataFrame,
    categorical_cols: list[str],
    continuous_cols: list[str],
    kind: str = "violin",
    height: float = 4.5,
    aspect: float = 1.2,
) -> None:
    """
    Generates a matrix layout grid comparing Categorical features against Continuous variables.
    Each subplot retains its individual X and Y axis labels and explicit titles for maximum clarity.

    kind: 'violin', 'box', or 'strip'
    """
    if not categorical_cols or not continuous_cols:
        print("Warning: Categorical or continuous column list is empty. Skipping matrix plot.")
        return

    n_rows = len(categorical_cols)
    n_cols = len(continuous_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * height * aspect, n_rows * height), squeeze=False
    )

    for i, cat in enumerate(categorical_cols):
        for j, cont in enumerate(continuous_cols):
            ax = axes[i, j]

            # Render the selected plot type
            if kind == "violin":
                sns.violinplot(data=df, x=cat, y=cont, ax=ax)
            elif kind == "box":
                sns.boxplot(data=df, x=cat, y=cont, ax=ax)
            elif kind == "strip":
                sns.stripplot(data=df, x=cat, y=cont, jitter=True, alpha=0.5, ax=ax)
            else:
                raise ValueError("kind must be 'violin', 'box', or 'strip'")

            # FORCE individual labels on every single subplot
            ax.set_xlabel(cat, fontsize=11, fontweight="bold")
            ax.set_ylabel(cont, fontsize=11, fontweight="bold")

            # ADD an individual title to each descriptive cell
            ax.set_title(f"Distribution of {cont} by {cat}", fontsize=10, pad=10)

            # ROTATE x-axis ticks slightly to prevent text overlapping across dense matrices
            ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> None:
    """Computes and renders a lower-triangle correlation heatmap for numeric parameters."""
    if not cols:
        print("Warning: No columns provided for correlation matrix plotting.")
        return

    corr = df[cols].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title(f"{method.capitalize()} Correlation (Lower Triangle)")
    plt.show()
