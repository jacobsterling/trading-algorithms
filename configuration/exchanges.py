from dotenv import load_dotenv
import os
from constants import PROJECT_ROOT
from configuration.execution import PAIR_WHITELIST, PAIR_BLACKLIST

load_dotenv(PROJECT_ROOT)


def get_exchange_config(exchange: str):
    match exchange:
        case "binance":
            return BINANCE_CONFIG
        case "kucoinfutures":
            return KUCOINFUTURES_CONFIG
        case "okx":
            return OKX_CONFIG
        case _:
            raise ValueError(f"Exchange {exchange} not configured")


_CCTX_CONFIG = {
    "ccxt_config": {"enableRateLimit": True, "futures": True},
    "ccxt_async_config": {"enableRateLimit": True, "futures": True},
}

BINANCE_CONFIG = {
    "name": "binance",
    "key": os.getenv("BINANCE_KEY"),
    "secret": os.getenv("BINANCE_SECRET"),
    "password": os.getenv("BINANCE_PASSWORD"),
    "pair_whitelist": PAIR_WHITELIST,
    "pair_blacklist": PAIR_BLACKLIST,
    "use_public_trades": True,
    **_CCTX_CONFIG,
}


KUCOINFUTURES_CONFIG = {
    "name": "kucoinfutures",
    "key": os.getenv("KUCOIN_KEY"),
    "secret": os.getenv("KUCOIN_SECRET"),
    "password": os.getenv("KUCOIN_PASSWORD"),
    "pair_whitelist": PAIR_WHITELIST,
    "pair_blacklist": PAIR_BLACKLIST,
    "use_public_trades": True,
    **_CCTX_CONFIG,
}

OKX_CONFIG = {
    "name": "okx",
    "key": os.getenv("OKX_KEY"),
    "secret": os.getenv("OKX_SECRET"),
    "password": os.getenv("OKX_PASSWORD"),
    "pair_whitelist": PAIR_WHITELIST,
    "pair_blacklist": PAIR_BLACKLIST,
    **_CCTX_CONFIG,
}
