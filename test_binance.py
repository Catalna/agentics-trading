import ccxt
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "2Ye00jLG1AxiSx3I2vSOdK6QUMkbfcJGygZmafERIiHKk97zUcYKrPs2GAafbTXm"
API_SECRET = "wHSip7tLnKqzmLxilr4tmjuS75hUVMHp3sozBx73JYV6jJlbC2ntN6pUvhpWnxb9"

print("--- TESTING METHOD 1: MANUAL URL OVERRIDE ---")
ex1 = ccxt.binanceusdm({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "options": {"defaultType": "future"},
    "verify": False,
})
ex1.urls['api']['fapiPublic'] = 'https://testnet.binancefuture.com/fapi/v1'
ex1.urls['api']['fapiPrivate'] = 'https://testnet.binancefuture.com/fapi/v1'
ex1.urls['api']['fapiPublicV2'] = 'https://testnet.binancefuture.com/fapi/v2'
ex1.urls['api']['fapiPrivateV2'] = 'https://testnet.binancefuture.com/fapi/v2'

try:
    bal = ex1.fetch_balance()
    print("Method 1 Success! USDT Balance:", bal.get("USDT", {}).get("total", 0.0))
except Exception as e:
    print("Method 1 Failed:", e)


print("\n--- TESTING METHOD 2: SET SANDBOX TRUE (WITH CCXT TESTNET URLS) ---")
ex2 = ccxt.binanceusdm({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "options": {"defaultType": "future"},
    "verify": False,
})
ex2.set_sandbox_mode(True)
# Override URLs to legacy testnet just in case CCXT's default sandbox domain is blocked/deprecated
ex2.urls['api']['fapiPublic'] = 'https://testnet.binancefuture.com/fapi/v1'
ex2.urls['api']['fapiPrivate'] = 'https://testnet.binancefuture.com/fapi/v1'
ex2.urls['api']['fapiPublicV2'] = 'https://testnet.binancefuture.com/fapi/v2'
ex2.urls['api']['fapiPrivateV2'] = 'https://testnet.binancefuture.com/fapi/v2'

try:
    bal = ex2.fetch_balance()
    print("Method 2 Success! USDT Balance:", bal.get("USDT", {}).get("total", 0.0))
except Exception as e:
    print("Method 2 Failed:", e)
