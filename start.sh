#!/bin/bash

# Set default exchange if not provided
EXCHANGE=${EXCHANGE:-binance}

# Generate config
echo "Generating config for exchange: $EXCHANGE"
python /freqtrade/main.py $EXCHANGE

# echo "Starting Freqtrade"
# freqtrade trade --config /freqtrade/config.json --strategy CVDDivergence

echo "Running backtesting"
freqtrade backtesting --strategy CVDDivergence --timerange 20240901-20240930 --timeframe 5m --max-open-trades 2 -p BTC/USDT:USDT --datadir /freqtrade/data/binance

# --enable-protections

# echo "Running hyperopt"
# freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --spaces buy --strategy CVDDivergence --datadir /freqtrade/data/binance --max-open-trades 2 --timeframe 5m --timerange 20240922-20240930 -p BTC/USDT:USDT

# echo "Downloading data"
# freqtrade download-data --exchange $EXCHANGE --pairs BTC/USDT:USDT --timeframe 1m --timerange 20240923-20240927 --trading-mode futures --datadir /freqtrade/data/binance

# echo "Downloading trade data"
# freqtrade download-data -p BTC/USDT:USDT --timerange 20240901-20240906 --trading-mode futures --timeframes 5m --dl-trades --exchange binance -v --prepend

# freqtrade download-data -p BTC/USDT:USDT --timerange 20240907-20240913 --trading-mode futures --timeframes 5m --dl-trades --exchange binance -v --prepend

# freqtrade download-data -p BTC/USDT:USDT --timerange 20240914-20240920 --trading-mode futures --timeframes 5m --dl-trades --exchange binance -v --prepend

# freqtrade download-data -p BTC/USDT:USDT --timerange 20240921-20240927 --trading-mode futures --timeframes 5m --dl-trades --exchange binance -v --prepend

# freqtrade download-data -p BTC/USDT:USDT --timerange 20240928-20241004 --trading-mode futures --timeframes 5m --dl-trades --exchange binance -v --prepend

