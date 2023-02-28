import argparse
import pandas
import logger

from src.constants import relevant_attributes
from src.nn import make_neural_network
from src.scale import scale
from src.visualization import visualize_data_details


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


def main():
    log = logger.get_logger('pretprocessor')
    args = get_args()
    log.info(f"Starting preprocessor for {args.path}")
    df = read_csv(args.path)

    ### Graphical data representation
    if args.visualize:
        visualize_data_details(df)

    ### Replace NaN with 0
    df = df.fillna(0)

    ### Normalize data
    scale(df, args.path)

    ### Feed it to NN
    make_neural_network(df)

    output_path = args.path.replace('input', 'output')
    df.to_csv(output_path)
    log.info(f"Preprocessing finished and dumped to {output_path} ")


if __name__ == "__main__":
    main()
