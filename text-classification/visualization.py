import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Optional, Union


def plot_barplot(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Optional[tuple] = (12, 4),
    xticks_rotation: Optional[int] = 0,
) -> None:
    """
    Plot a bar plot showing the value counts of a categorical or boolean Series.

    Args:
        data (pd.Series): Categorical or boolean data to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (Optional[tuple], optional): Size of the figure as (width, height). Defaults to (12, 4).
        xticks_rotation (Optional[int], optional): X-ticks rotation on the plot. Defaults to 0.

    Returns:
        None: The function directly modifies the plot and does not return any values.
    """

    plt.figure(figsize=figsize)

    # get value counts of categorical values
    category_counts = data.value_counts()

    # generate barplot
    ax = sns.barplot(x=category_counts.index, y=category_counts.values)

    # add count on top of each bar
    for i, count in enumerate(category_counts.values):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=10)

    # set plot title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=xticks_rotation)

    # display grid for better readability and show the barplot
    plt.grid(True)
    plt.show()


def plot_histogram(
    x: Union[pd.Series, np.ndarray, list],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Optional[tuple] = (10, 6),
    bins: Optional[Union[list, np.ndarray]] = None,
    generate_bins: Optional[bool] = False,
    num_bins: Optional[int] = 20,
) -> None:
    """
     Plots a histogram with optional custom bins, labels, and grid.

    Args:
        x (Union[pd.Series, np.ndarray, list]): The data to be plotted in the histogram.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        figsize (Optional[tuple], optional): The size of the figure (width, height). Default is (10, 6).
        bins (Optional[Union[list, np.ndarray]], optional): Custom bin edges. Default is None.
        generate_bins (Optional[bool], optional): If True, generates evenly spaced bins based on the data range. Default is False.
        num_bins (Optional[int], optional): The number of bins to generate if `generate_bins` is True. Default is 20.

    Returns:
        None: This function directly modifies the plot and does not return any values.
    """

    plt.figure(figsize=figsize)

    # generate evenly spaced integer bins if requested
    if generate_bins:
        step_bins = (max(x) - min(x)) // num_bins
        bins = np.arange(min(x), max(x) + step_bins, step_bins)

    # plot histogram with specified or generated bins
    if isinstance(bins, (np.ndarray, list)):
        counts, bins, _ = plt.hist(
            x, bins=bins, color="cornflowerblue", edgecolor="red", alpha=1
        )
        plt.xticks(ticks=bins, rotation=45)
    # plot histogram without specified bins
    else:
        counts, bins, _ = plt.hist(x, color="cornflowerblue", edgecolor="red", alpha=1)

    # plot count number on top of each bin
    for i in range(len(counts)):
        plt.text(
            x=(bins[i] + bins[i + 1]) / 2,
            y=counts[i],
            s=int(counts[i]),
            ha="center",
            va="bottom",
        )

    # set labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # display grid for better readability and show the histogram
    plt.grid(True)
    plt.show()


def plot_word_cloud(
    wordcloud: WordCloud, title: str, figsize: Optional[tuple] = (12, 6)
) -> None:
    """
    Plot a word cloud.

    Args:
        wordcloud (WordCloud): The word cloud object to be plotted.
        title (str): The title of the plot.
        figsize (Optional[tuple], optional): Size of the figure as (width, height). Defaults to (12, 6).
    """

    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()
