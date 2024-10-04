from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from constants import DATA_DIR
from pandas import DataFrame


def load_test_data(
    pair, timeframe, candle_type: CandleType = CandleType.FUTURES
) -> DataFrame:

    return load_pair_history(
        datadir=DATA_DIR,
        timeframe=timeframe,
        pair=pair,
        data_format="feather",
        candle_type=candle_type,
    )
