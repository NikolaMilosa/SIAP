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


def scale_btc(df):
    standard_scale(df, "BlkCnt")
    min_max_scale(df, "CapMrktCurUSD")
    min_max_scale(df, "DiffMean")
    robust_scale(df, "FeeMeanUSD")
    robust_scale(df, "FlowInExUSD")
    min_max_scale(df, "HashRate")
    robust_scale(df, "ROI30d")


def scale_eth(df):
    robust_scale(df, "BlkCnt")
    min_max_scale(df, "CapMrktCurUSD")
    min_max_scale(df, "DiffMean")
    robust_scale(df, "FeeMeanUSD")
    robust_scale(df, "FlowInExUSD")
    min_max_scale(df, "HashRate")
    robust_scale(df, "ROI30d")


def scale(df, path):
    log = logger.get_logger('scaler')
    if ScaleOptions.BTC.value in path:
        scale_btc(df)
        log.info("Bitcoin dataframe normalized successfully.")
        return
    if ScaleOptions.ETH.value in path:
        scale_eth(df)
        log.info("Etherium dataframe normalized successfully.")
        return
    log.warn("Normalizer for provided crypto coin doesn't exist!")
