import argparse
import pandas
import logger
import seaborn as sns
import matplotlib.pyplot as plt

relevant_attributes = ["time", "BlkCnt", "CapMrktCurUSD", "DiffMean", "FeeMeanUSD", "FlowInExUSD", "HashRate", "NDF",
                       "ROI30d", "PriceUSD"]


def get_args():
    parser = argparse.ArgumentParser(prog="Preprocessor for datasets for crypto")
    parser.add_argument("path", type=str, help="Path to the dataset csv which should be loaded")

    return parser.parse_args()


def read_csv(path):
    df = pandas.read_csv(rf'{path}',
                         usecols=relevant_attributes
                         )
    return df


def plot_df_columns(df):
    sns.pairplot(df)
    plt.show()


def main():
    log = logger.get_logger('pretprocessor')
    args = get_args()
    log.info(f"Starting preprocessor for {args.path}")
    df = read_csv(args.path)

    ### Uncomment for graphical data representation
    # plot_df_columns(df)

    ### Replace NaN with 0
    df = df.fillna(0)

    df.to_csv(args.path)
    log.info(f"Preprocessing finished and dumped to {args.path} ")


if __name__ == "__main__":
    main()
