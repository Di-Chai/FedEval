from Crypto.Protocol.KDF import PBKDF2
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# pycryptodome


# Authenticated encryption on a string using AES GCM with both encryption and MAC
def aes_gcm_encrypt(private_key, message, user_add='Default Additional Information'):
    kdf_salt = get_random_bytes(16)
    private_key = PBKDF2(private_key, kdf_salt)
    cipher = AES.new(private_key, AES.MODE_GCM)
    cipher.update(user_add.encode())
    ciphertext, tag = cipher.encrypt_and_digest(message.encode())
    return user_add, ciphertext, tag, cipher.nonce, kdf_salt


def aes_gcm_decrypt(key, received_message):
    received_aad, received_ciphertext, received_tag, received_nonce, received_kdf_salt = received_message
    key = PBKDF2(key, received_kdf_salt)
    cipher = AES.new(key, AES.MODE_GCM, received_nonce)
    cipher.update(received_aad.encode())
    try:
        message = cipher.decrypt_and_verify(received_ciphertext, received_tag).decode()
    except ValueError:
        message = None
        print("\nMAC validation failed during decryption. No authentication gurantees on this ciphertext")
        print("\nUnauthenticated AAD: " + str(received_aad))
    return received_aad, message
