import pandas
import logger

from src.constants import relevant_attributes
from src.nn import create_neural_network
from src.preprocess import preprocess
from args import get_args
from src.visualization import visualize_data_details


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
    preprocess(df, args.path)

    ### Feed it to NN
    create_neural_network(df, args.path)


if __name__ == "__main__":
    main()
