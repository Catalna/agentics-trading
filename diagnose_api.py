import os
import ccxt
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("=" * 60)
print("       BINANCE API KEY DIAGNOSTIC UTILITY")
print("=" * 60)

# Load .env
if os.path.exists(".env"):
    load_dotenv()
    print("[+] Loaded .env file successfully.")
else:
    print("[-] .env file not found in current directory!")

api_key = os.getenv("BINANCE_API_KEY", "").strip()
api_secret = os.getenv("BINANCE_API_SECRET", "").strip()

print(f"[i] API Key: '{api_key[:6]}...{api_key[-6:]}' (Length: {len(api_key)})")
print(f"[i] API Secret: '{api_secret[:6]}...{api_secret[-6:]}' (Length: {len(api_secret)})")

if not api_key or not api_secret:
    print("[-] ERROR: API Key or Secret is empty in .env!")
    exit(1)

def test_environment(name, urls_override=None, enable_demo=False):
    print(f"\n>>> Testing {name}...")
    try:
        ex = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "options": {"defaultType": "future"},
            "verify": False,
            "enableRateLimit": True
        })
        if enable_demo:
            try:
                ex.enable_demo_trading(True)
            except Exception as e:
                print(f"    [i] Demo trading enabling failed: {e}")
                
        if urls_override:
            for k, v in urls_override.items():
                ex.urls['api'][k] = v
                
        # Try a public request first
        try:
            ex.fetch_ohlcv("BTC/USDT", "5m", limit=1)
            print("    [+] Public Connection: SUCCESS (Able to reach server)")
        except Exception as e:
            print(f"    [-] Public Connection: FAILED ({type(e).__name__}: {e})")
            return
            
        # Try set_leverage (sometimes doesn't fail immediately on invalid keys, depending on exchange cache/mocking)
        try:
            ex.set_leverage(25, "BTC/USDT")
            print("    [+] Private set_leverage: SUCCESS")
        except Exception as e:
            print(f"    [-] Private set_leverage: FAILED ({type(e).__name__}: {e})")
            
        # Try fetch_balance (strict signing & permissions check)
        try:
            bal = ex.fetch_balance()
            usdt_bal = bal.get("USDT", {}).get("total", 0.0)
            print(f"    [+] Private fetch_balance: SUCCESS! USDT Balance: {usdt_bal}")
        except Exception as e:
            print(f"    [-] Private fetch_balance: FAILED ({type(e).__name__}: {e})")
            
    except Exception as e:
        print(f"    [-] Unexpected setup error: {e}")

# 1. Test Legacy Testnet (unblocked)
test_environment(
    "1. Legacy Testnet (testnet.binancefuture.com)",
    urls_override={
        "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
        "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
        "fapiPublicV2": "https://testnet.binancefuture.com/fapi/v2",
        "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2"
    }
)

# 2. Test New Demo Portal (usually blocked by ISP)
test_environment(
    "2. New Demo Trading (demo-fapi.binance.com)",
    enable_demo=True
)

# 3. Test Binance Mainnet (real funds)
test_environment(
    "3. Real Mainnet (fapi.binance.com)"
)

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETED.")
print("Please copy and paste the entire terminal output back to Antigravity!")
print("=" * 60)
