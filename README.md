# TLS Forensics: AI-assisted TLS Key Material Detection from Memory

This project implements a memory forensics tool to identify TLS key material using machine learning.

## Features
- Random Forest model with entropy and statistical feature extraction
- Pattern matching for TLS 1.2 and 1.3
- Cryptographic validation using OpenSSL
- Tamper-evident forensic logging
- Cross-platform memory dump analysis (Linux, Windows)

## How to Use
```bash
python main.py --dump memory.dmp
```

## License
MIT License
