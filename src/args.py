import argparse

parser = argparse.ArgumentParser(prog="Preprocessor for datasets for crypto")
parser.add_argument("path", type=str, help="Path to the dataset csv which should be loaded")
parser.add_argument("visualize", type=bool, help="Should dataset be visualized")
parser.add_argument("num_epochs", type=int, help="Number of epochs to train")

def get_args():
    return parser.parse_args()