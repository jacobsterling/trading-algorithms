import pytest
import pandas as pd
import numpy as np
from constants import CONFIG_DIR
from tests.utils import load_test_data
from freqtrade.data.dataprovider import DataProvider
from freqtrade.resolvers import StrategyResolver
from freqtrade.configuration import Configuration


@pytest.fixture
def pair():
    return "BTC/USDT:USDT"


@pytest.fixture
def timeframe():
    return "5m"


@pytest.fixture
def strategy(pair, timeframe):
    return "CVDDivergence"


@pytest.fixture
def config(timeframe, strategy):

    cfg = Configuration.from_files([CONFIG_DIR])

    cfg.update(
        {
            "timeframe": timeframe,
            "strategy": strategy,
        }
    )

    return cfg


@pytest.fixture
def strategy_instance(config):

    strategy = StrategyResolver.load_strategy(config)

    strategy.dp = DataProvider(config, None, None)
    strategy.ft_bot_start()

    return strategy


@pytest.fixture
def candles(pair, timeframe):
    return load_test_data(pair, timeframe)


@pytest.fixture
def analyzed_candles(strategy_instance, candles, pair):
    return strategy.analyze_ticker(candles, {"pair": pair})
