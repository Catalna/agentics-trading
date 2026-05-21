import ccxt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

ex = ccxt.binanceusdm({
    "enableRateLimit": True,
    "verify": False
})
ex.set_sandbox_mode(True)
ex.load_markets()
ob = ex.fetch_order_book("BTC/USDT:USDT", limit=5)
print("Order Book fetched successfully!")
print("Bids (Top 3):", ob['bids'][:3])
print("Asks (Top 3):", ob['asks'][:3])

