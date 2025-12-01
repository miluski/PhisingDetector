#!/bin/bash

CERT_DIR="certs"
KEY_FILE="$CERT_DIR/key.pem"
CERT_FILE="$CERT_DIR/cert.pem"

echo "==============================================="
echo "  HTTPS Certificate Generation Script"
echo "==============================================="
echo ""

mkdir -p "$CERT_DIR"

if [ -f "$KEY_FILE" ] && [ -f "$CERT_FILE" ]; then
    echo "Certificates already exist in $CERT_DIR/"
    echo ""
    read -p "Do you want to regenerate them? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing certificates."
        exit 0
    fi
    echo "Regenerating certificates..."
fi

echo "Generating self-signed SSL certificate..."
echo ""

openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -days 365 \
    -subj "/C=PL/ST=State/L=City/O=PhishingDetector/CN=localhost"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Certificates generated successfully!"
    echo ""
    echo "Certificate files:"
    echo "  - Private key: $KEY_FILE"
    echo "  - Certificate:  $CERT_FILE"
    echo ""
    echo "Valid for: 365 days"
    echo ""
    echo "Note: These are self-signed certificates for development."
    echo "Browsers will show a security warning. This is expected."
    echo ""
else
    echo ""
    echo "✗ Error: Failed to generate certificates."
    echo "Please ensure OpenSSL is installed."
    exit 1
fi
