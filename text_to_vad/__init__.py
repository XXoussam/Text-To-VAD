"""DeBERTa-v3 Valence / Arousal / Dominance regression on EmoBank."""

# Verify HTTPS with the OS certificate store instead of certifi, so Hugging Face
# downloads work behind corporate proxies that re-sign TLS traffic. Must run
# before `requests` / `urllib3` open any connection.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass
