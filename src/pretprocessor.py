import argparse
import pandas
import logger
import json

from src.constants import relevant_attributes
from src.nn import make_neural_network
from src.scale import scale
from src.visualization import visualize_data_details
from args import get_args


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
    # if args.visualize:
    #     visualize_data_details(df)

    ### Replace NaN with 0
    df = df.fillna(0)

    ### Normalize data
    scale(df, args.path)

    ### Feed it to NN
    result = make_neural_network(df)

    output_path = args.path.replace('input', 'output').replace('.csv', '.json')
    with open(output_path, 'w') as f:
        json.dump(result, f)
    df.to_csv(output_path)
    log.info(f"Preprocessing finished and dumped to {output_path} ")


if __name__ == "__main__":
    main()
