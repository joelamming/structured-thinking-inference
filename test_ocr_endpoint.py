#!/usr/bin/env python3
"""
Test script for /ws/ocr endpoint

This script validates that the OCR endpoint is working correctly with:
1. WebSocket connection
2. Authentication
3. Encryption/decryption
4. Request/response flow
5. Keepalive handling

Usage:
    python test_ocr_endpoint.py --url wss://your-instance/ws/ocr --api-key YOUR_KEY --encryption-key YOUR_FERNET_KEY --image path/to/image.png
"""

import asyncio
import argparse
import base64
import json
import sys
from pathlib import Path
from cryptography.fernet import Fernet
import websockets  # type: ignore[import-untyped]


class OCREndpointTester:
    def __init__(self, ws_url: str, api_key: str, encryption_key: str):
        self.ws_url = ws_url
        self.api_key = api_key
        self.fernet = Fernet(
            encryption_key.encode()
            if isinstance(encryption_key, str)
            else encryption_key
        )

    def encrypt(self, text: str) -> str:
        """Encrypt text and return Fernet token."""
        return self.fernet.encrypt(text.encode()).decode("ascii")

    def decrypt(self, token: str) -> str:
        """Decrypt Fernet token and return text."""
        return self.fernet.decrypt(token.encode()).decode("utf-8")

    def image_to_data_url(self, image_path: Path) -> str:
        """Convert image file to data URL."""
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("ascii")

        # Detect MIME type
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(image_path.suffix.lower(), "image/png")

        return f"data:{mime_type};base64,{img_data}"

    def build_request(
        self, image_path: Path, prompt: str = "Convert the document to markdown."
    ) -> dict:
        """Build an encrypted OCR request."""
        data_url = self.image_to_data_url(image_path)

        return {
            "request": {
                "model": "deepseek-ai/DeepSeek-OCR",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": self.encrypt(data_url)},
                            },
                            {"type": "text", "text": self.encrypt(prompt)},
                        ],
                    }
                ],
                "max_tokens": 8000,
                "temperature": 0.6,
            },
            "client_request_id": f"test-{image_path.stem}",
            "timeout_seconds": 300,
        }

    async def test_connection(self):
        """Test 1: WebSocket connection and authentication."""
        print("Test 1: WebSocket Connection & Authentication")
        try:
            headers = {"X-API-Key": self.api_key}
            async with websockets.connect(  # type: ignore[attr-defined]
                self.ws_url, extra_headers=headers, open_timeout=10
            ):
                print("  ✓ Connection established")
                print("  ✓ Authentication successful")
                return True
        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 403:
                print("  ✗ Authentication failed (403)")
            else:
                print(f"  ✗ Connection failed: {e}")
            return False
        except Exception as e:
            print(f"  ✗ Connection error: {e}")
            return False

    async def test_keepalive(self):
        """Test 2: Keepalive ping/pong."""
        print("\nTest 2: Keepalive Ping/Pong")
        try:
            headers = {"X-API-Key": self.api_key}
            async with websockets.connect(self.ws_url, extra_headers=headers) as ws:
                # Wait for ping
                print("  Waiting for ping (up to 20 seconds)...")
                msg = await asyncio.wait_for(ws.recv(), timeout=20)
                data = json.loads(msg)

                if data.get("type") == "ping":
                    print("  ✓ Received ping")

                    # Send pong
                    await ws.send(json.dumps({"type": "pong"}))
                    print("  ✓ Sent pong")
                    return True
                else:
                    print(f"  ? Unexpected message: {data}")
                    return False
        except asyncio.TimeoutError:
            print("  ✗ No ping received within 20 seconds")
            return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    async def test_ocr_request(self, image_path: Path):
        """Test 3: Full OCR request/response cycle."""
        print("\nTest 3: OCR Request/Response")
        try:
            headers = {"X-API-Key": self.api_key}
            async with websockets.connect(self.ws_url, extra_headers=headers) as ws:  # type: ignore[attr-defined]
                # Build and send request
                request = self.build_request(image_path)
                print(f"  Sending OCR request for: {image_path.name}")
                await ws.send(json.dumps(request))
                print("  ✓ Request sent")

                # Handle pings while waiting for response
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=120)
                    data = json.loads(msg)

                    if data.get("type") == "ping":
                        await ws.send(json.dumps({"type": "pong"}))
                        print("  → Handled ping")
                        continue

                    # Got response
                    if data.get("status") == "completed":
                        print("  ✓ Received completed response")

                        # Decrypt content
                        encrypted_content = data["result"]["choices"][0]["message"][
                            "content"
                        ]
                        decrypted = self.decrypt(encrypted_content)

                        print("  ✓ Successfully decrypted response")
                        print("\n  Markdown Output (first 200 chars):")
                        print(f"  {decrypted[:200]}...")

                        # Check usage stats
                        usage = data["result"].get("usage", {})
                        print(f"\n  Tokens: {usage.get('total_tokens', 'N/A')}")

                        return True
                    elif data.get("status") in ["error", "timeout"]:
                        print(f"  ✗ Request failed: {data.get('error')}")
                        return False
                    else:
                        print(f"  ? Unexpected response: {data}")
                        return False

        except asyncio.TimeoutError:
            print("  ✗ Timeout waiting for response")
            return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def test_encryption(self):
        """Test 4: Encryption/decryption roundtrip."""
        print("\nTest 4: Encryption/Decryption")
        try:
            test_strings = [
                "Hello, World!",
                "data:image/png;base64,iVBORw0KGgo...",
                "Convert this document to markdown.",
            ]

            for test_str in test_strings:
                encrypted = self.encrypt(test_str)
                decrypted = self.decrypt(encrypted)

                if decrypted == test_str:
                    print(f"  ✓ Roundtrip successful for: {test_str[:30]}...")
                else:
                    print("  ✗ Roundtrip failed!")
                    return False

            return True
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    async def run_all_tests(self, image_path: Path | None = None):
        """Run all tests."""
        print("=" * 60)
        print("OCR Endpoint Test Suite")
        print("=" * 60)

        results = []

        # Test 1: Connection
        results.append(await self.test_connection())

        # Test 2: Keepalive
        results.append(await self.test_keepalive())

        # Test 3: Encryption
        results.append(await self.test_encryption())

        # Test 4: OCR Request (if image provided)
        if image_path and image_path.exists():
            results.append(await self.test_ocr_request(image_path))
        else:
            print("\nTest 4: OCR Request - SKIPPED (no image provided)")

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        passed = sum(results)
        total = len(results)
        print(f"Passed: {passed}/{total}")

        if passed == total:
            print("✓ All tests passed!")
            return 0
        else:
            print("✗ Some tests failed")
            return 1


async def main():
    parser = argparse.ArgumentParser(description="Test /ws/ocr endpoint")
    parser.add_argument(
        "--url", required=True, help="WebSocket URL (e.g., wss://host/ws/ocr)"
    )
    parser.add_argument("--api-key", required=True, help="API key (ORCH_API_KEY)")
    parser.add_argument("--encryption-key", required=True, help="Fernet encryption key")
    parser.add_argument("--image", type=Path, help="Path to test image (optional)")

    args = parser.parse_args()

    # Validate image if provided
    if args.image and not args.image.exists():
        print(f"Error: Image file not found: {args.image}")
        return 1

    # Run tests
    tester = OCREndpointTester(args.url, args.api_key, args.encryption_key)
    return await tester.run_all_tests(args.image)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
