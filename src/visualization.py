import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy

from src.constants import output_attribute, relevant_attributes, histogram_bin_num, histograms_bin_num


def plot_df_columns(df):
    sns.pairplot(df, y_vars=output_attribute, x_vars=relevant_attributes)
    plt.show()


def print_description(df):
    numpy.set_printoptions(threshold=sys.maxsize)
    ### Keeps output decimal
    # print_val = df.describe().apply(lambda s: s.apply('{0:.2f}'.format))
    print_val = df.describe().round(2)
    print(print_val.to_string())


def plot_histograms(df, bin_num):
    for attribute in relevant_attributes:
        if attribute != "time":
            df[attribute].plot(kind='hist', bins=bin_num, title=attribute)
            plt.show()


def plot_histogram(df, bin_num):
    df.hist(bins=bin_num)
    plt.show()


def visualize_data_details(df):
    plot_df_columns(df)
    plot_histogram(df, histogram_bin_num)
    plot_histograms(df, histograms_bin_num)
    print_description(df)