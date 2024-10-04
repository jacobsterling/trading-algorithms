#!/bin/bash

# Set default exchange if not provided
EXCHANGE=${EXCHANGE:-binance}

# Generate config
echo "Generating config for exchange: $EXCHANGE"
python /freqtrade/main.py $EXCHANGE

# echo "Starting Freqtrade"
# freqtrade trade --config /freqtrade/config.json --strategy CVDDivergence

# echo "Running backtesting"
# freqtrade backtesting --strategy CVDDivergence --timerange 20240901-20240930 --timeframe 5m --max-open-trades 2 -p BTC/USDT:USDT --enable-protections --datadir /freqtrade/data/binance

# echo "Running hyperopt"
# freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --spaces buy --strategy CVDDivergence --datadir /freqtrade/data/binance --max-open-trades 2 --timeframe 1m --timerange 20240923-20240927

# echo "Downloading data"
# freqtrade download-data --exchange $EXCHANGE --pairs BTC/USDT:USDT --timeframe 1m --timerange 20240923-20240927 --trading-mode futures --datadir /freqtrade/data/binance

echo "Downloading trade data"
freqtrade download-data -p BTC/USDT:USDT --timerange 20240909-20240916 --trading-mode futures --timeframes 5m --dl-trades --exchange binance -v --prepend
