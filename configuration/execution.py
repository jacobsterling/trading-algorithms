ENTRY_PRICING = {
    "price_side": "same",
    "use_order_book": True,
    "order_book_top": 1,
    "price_last_balance": 0.0,
    "check_depth_of_market": {"enabled": False, "bids_to_ask_delta": 1},
}

EXIT_PRICING = {
    "price_side": "same",
    "use_order_book": True,
    "order_book_top": 1,
}

UNFILLED_TIMEOUT = {
    "entry": 10,
    "exit": 10,
    "exit_timeout_count": 0,
    "unit": "minutes",
}

DRY_RUN = True
DRY_RUN_WALLET = 1000

_TRADING_MODE = {
    "trading_mode": "futures",
    "margin_mode": "isolated",
}

ORDERFLOW = {
    "cache_size": 1000,
    "max_candles": 1500,
    "scale": 0.5,
    "stacked_imbalance_range": 3,  # needs at least this amount of imbalance next to each other
    "imbalance_volume": 1,  # filters out below
    "imbalance_ratio": 3,  # filters out ratio lower than
}


PAIR_WHITELIST = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
PAIR_BLACKLIST = []

RESTRICTIONS = {
    "max_open_trades": 3,
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.99,
    "dry_run": False,
    "cancel_open_orders_on_exit": False,
    "force_entry_enable": False,
    "order_book_max": 5,
    "order_book_min": 1,
    **_TRADING_MODE,
    "orderflow": ORDERFLOW,
}

if DRY_RUN:
    RESTRICTIONS.update({"dry_run": True, "dry_run_wallet": DRY_RUN_WALLET})
