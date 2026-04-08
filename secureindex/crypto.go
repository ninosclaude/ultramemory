package secureindex

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/hkdf"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"strings"
)

const encryptedPrefix = "enc:gcm:"

// IsEncrypted reports whether value uses the secureindex ciphertext envelope.
func IsEncrypted(value string) bool {
	return strings.HasPrefix(value, encryptedPrefix)
}

// EncryptString seals plaintext with an AES-GCM key derived from the personal key and purpose.
func EncryptString(personalKey, purpose, plaintext string) (string, error) {
	if plaintext == "" {
		return "", nil
	}
	if personalKey == "" {
		return "", fmt.Errorf("secureindex: missing personal key for %s", purpose)
	}

	blockKey, err := deriveKey(personalKey, purpose)
	if err != nil {
		return "", err
	}

	block, err := aes.NewCipher(blockKey)
	if err != nil {
		return "", fmt.Errorf("secureindex: create cipher: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", fmt.Errorf("secureindex: create gcm: %w", err)
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", fmt.Errorf("secureindex: read nonce: %w", err)
	}

	ciphertext := gcm.Seal(nil, nonce, []byte(plaintext), nil)
	out := append(nonce, ciphertext...)
	return encryptedPrefix + base64.StdEncoding.EncodeToString(out), nil
}

// DecryptString opens a secureindex ciphertext envelope.
// Plaintext legacy values pass through unchanged to keep old databases readable.
func DecryptString(personalKey, purpose, value string) (string, error) {
	if value == "" {
		return "", nil
	}
	if !IsEncrypted(value) {
		return value, nil
	}
	if personalKey == "" {
		return "", fmt.Errorf("secureindex: missing personal key for %s", purpose)
	}

	raw, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(value, encryptedPrefix))
	if err != nil {
		return "", fmt.Errorf("secureindex: decode ciphertext: %w", err)
	}

	blockKey, err := deriveKey(personalKey, purpose)
	if err != nil {
		return "", err
	}

	block, err := aes.NewCipher(blockKey)
	if err != nil {
		return "", fmt.Errorf("secureindex: create cipher: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", fmt.Errorf("secureindex: create gcm: %w", err)
	}
	if len(raw) < gcm.NonceSize() {
		return "", fmt.Errorf("secureindex: ciphertext too short")
	}

	nonce := raw[:gcm.NonceSize()]
	ciphertext := raw[gcm.NonceSize():]
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return "", fmt.Errorf("secureindex: decrypt %s: %w", purpose, err)
	}
	return string(plaintext), nil
}

// LookupDigest returns a stable keyed digest for equality lookups without storing plaintext.
func LookupDigest(personalKey, purpose, plaintext string) (string, error) {
	if personalKey == "" {
		return "", fmt.Errorf("secureindex: missing personal key for %s", purpose)
	}
	key, err := deriveKey(personalKey, purpose)
	if err != nil {
		return "", err
	}
	sum := sha256.New()
	if _, err := sum.Write(key); err != nil {
		return "", fmt.Errorf("secureindex: hash key %s: %w", purpose, err)
	}
	if _, err := sum.Write([]byte(strings.ToLower(strings.TrimSpace(plaintext)))); err != nil {
		return "", fmt.Errorf("secureindex: hash plaintext %s: %w", purpose, err)
	}
	return hex.EncodeToString(sum.Sum(nil)), nil
}

func deriveKey(personalKey, purpose string) ([]byte, error) {
	info := "ultramemory/" + purpose
	key, err := hkdf.Key(sha256.New, []byte(personalKey), nil, info, 32)
	if err != nil {
		return nil, fmt.Errorf("secureindex: hkdf %s: %w", purpose, err)
	}
	return key, nil
}
