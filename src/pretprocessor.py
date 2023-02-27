import argparse
import pandas
import logger
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy

relevant_attributes = ["time", "BlkCnt", "CapMrktCurUSD", "DiffMean", "FeeMeanUSD", "FlowInExUSD", "HashRate", "NDF",
                       "ROI30d", "PriceUSD"]

output_attribute = "PriceUSD"


def get_args():
    parser = argparse.ArgumentParser(prog="Preprocessor for datasets for crypto")
    parser.add_argument("path", type=str, help="Path to the dataset csv which should be loaded")
    parser.add_argument("visualize", type=bool, help="Should dataset be visualized")

    return parser.parse_args()


def read_csv(path):
    df = pandas.read_csv(rf'{path}',
                         usecols=relevant_attributes
                         )
    return df


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


def main():
    log = logger.get_logger('pretprocessor')
    args = get_args()
    log.info(f"Starting preprocessor for {args.path}")
    df = read_csv(args.path)

    ### Replace NaN with 0
    df = df.fillna(0)

    ### Graphical data representation
    if args.visualize:
        plot_df_columns(df)
        plot_histogram(df, 40)
        plot_histograms(df, 100)
        print_description(df)

    output_path = args.path.replace('input', 'output')
    df.to_csv(output_path)
    log.info(f"Preprocessing finished and dumped to {output_path} ")


if __name__ == "__main__":
    main()
