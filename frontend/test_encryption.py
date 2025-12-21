#!/usr/bin/env python3
"""
Test script to verify Fernet encryption/decryption
"""

import os
from cryptography.fernet import Fernet


def main():
    # Check if key exists
    key = os.getenv("ORCH_ENCRYPTION_KEY")

    if not key:
        print("⚠️  ORCH_ENCRYPTION_KEY not set")
        print("\nGenerating a new key for testing:")
        key = Fernet.generate_key().decode()
        print(f"\n{key}\n")
        print("Save this key and use it for both frontend and orchestrator:")
        print(f"export ORCH_ENCRYPTION_KEY='{key}'")
        return

    print(f"✓ Encryption key found: {key[:20]}...")

    # Test encryption/decryption
    try:
        fernet = Fernet(key.encode())

        test_message = "Hello, world! This is a test message."
        print(f"\nOriginal: {test_message}")

        encrypted = fernet.encrypt(test_message.encode()).decode()
        print(f"\nEncrypted: {encrypted[:50]}...")

        decrypted = fernet.decrypt(encrypted.encode()).decode()
        print(f"\nDecrypted: {decrypted}")

        if test_message == decrypted:
            print("\n✅ Encryption/decryption working correctly!")
        else:
            print("\n❌ Decryption mismatch!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your key is a valid Fernet key (32 bytes, base64-encoded)")


if __name__ == "__main__":
    main()
