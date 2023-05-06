import pandas
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import logger
from enum import Enum


class ScaleOptions(Enum):
    BTC = 'btc'
    ETH = 'eth'


def robust_scale(df, col_name):
    scaler = RobustScaler()
    scaled_col = scaler.fit_transform(df[[col_name]])
    df[col_name] = scaled_col


def standard_scale(df, col_name):
    scaler = StandardScaler()
    scaled_col = scaler.fit_transform(df[[col_name]])
    df[col_name] = scaled_col


def min_max_scale(df, col_name):
    scaler = MinMaxScaler()
    scaled_col = scaler.fit_transform(df[[col_name]])
    df[col_name] = scaled_col


def min_max_log_scale(df, col_name):
    scaler = MinMaxScaler()
    scaled_col = scaler.fit_transform(df[[col_name]])
    df[col_name] = np.log(scaled_col + 1)


def convert_from_date_to_float(df, col_name):
    df[col_name] = pandas.to_datetime(df[col_name])
    df[col_name] = (df[col_name] - df[col_name].min()).astype('timedelta64[s]').astype('int32').astype('float32')


def scale_btc(df):
    convert_from_date_to_float(df, "time")
    standard_scale(df, "BlkCnt")
    min_max_scale(df, "CapMrktCurUSD")
    min_max_scale(df, "DiffMean")
    robust_scale(df, "FeeMeanUSD")
    robust_scale(df, "FlowInExUSD")
    min_max_scale(df, "HashRate")
    robust_scale(df, "ROI30d")


def scale_eth(df):
    convert_from_date_to_float(df, "time")
    standard_scale(df, "time")
    robust_scale(df, "BlkCnt")
    min_max_scale(df, "CapMrktCurUSD")
    min_max_scale(df, "DiffMean")
    min_max_log_scale(df, "FeeMeanUSD")
    min_max_log_scale(df, "FlowInExUSD")
    min_max_scale(df, "HashRate")
    robust_scale(df, "ROI30d")


def serialize_data(df, path):
    output_path = path.replace('input', 'output')
    df.to_csv(output_path)
    return output_path


def preprocess(df, path):
    log = logger.get_logger('scaler')
    if ScaleOptions.BTC.value in path:
        scale_btc(df)
        output_path = serialize_data(df, path)
        log.info(f"Preprocessing Bitcoin finished and dumped to {output_path} ")
        return
    if ScaleOptions.ETH.value in path:
        scale_eth(df)
        output_path = serialize_data(df, path)
        log.info(f"Preprocessing Etherium finished and dumped to {output_path} ")
        return
    log.warn("Preprocessor for provided crypto coin doesn't exist!")
