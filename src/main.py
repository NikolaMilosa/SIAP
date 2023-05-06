import pandas
import logger

from constants import relevant_attributes
from nn import create_neural_network
from preprocess import preprocess
from args import get_args
from visualization import visualize_data_details


def read_csv(path):
    df = pandas.read_csv(rf'{path}',
                         usecols=relevant_attributes
                         )
    return df


def main():
    log = logger.get_logger('main')
    args = get_args()
    log.info(f"Starting preprocessor for {args.path}")
    df = read_csv(args.path)

    ### Graphical data representation
    # if args.visualize:
    #     visualize_data_details(df)

    ### Replace NaN with 0
    df = df.fillna(0)

    ### Normalize data
    preprocess(df, args.path)

    ### Graphical data representation
    # if args.visualize:
    #     visualize_data_details(df)

    ### Feed it to NN
    create_neural_network(df, args.path, args.num_epochs)


if __name__ == "__main__":
    main()
