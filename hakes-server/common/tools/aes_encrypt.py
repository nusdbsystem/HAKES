"""
Copyright 2024 The HAKES Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import base64
import json
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

DEFAULT_KEY = b"\x1f\x86\x6a\x3b\x65\xb6\xae\xea\xad\x57\x34\x53\xd1\x03\x8c\x01"
DEFAULT_IV = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
DEFAULT_AAD = b"\x00\x00\x00\x00"


def get_args():
    parser = argparse.ArgumentParser(description="aes enc/dec helper script")
    parser.add_argument(
        "--mode", required=True, choices={"enc", "dec", "nenc"}, help=""
    )
    parser.add_argument("--input_file", required=True, help="file to enc/dec")
    parser.add_argument(
        "--cipher_txt_encoding",
        required=True,
        help="format that cipher text is/will be encoded in",
        choices={"b64", "hex", "bin"},
    )
    parser.add_argument("--aes_data_file", help="json file of metadata to use in aes")
    parser.add_argument(
        "--output_file",
        help='file to write output, default being input file name suffixed with "_enc/_dec"',
    )
    return parser.parse_args()


def get_aes_meta(json_file):
    if not json_file:
        # print("Debug print: default key in base64: " + str(base64.b64encode(DEFAULT_KEY)))
        return DEFAULT_KEY, DEFAULT_IV, DEFAULT_AAD
    with open(json_file, "rb") as f:
        aes_json = json.load(f)
        # assuming the contents are base64 encoded
        key = base64.b64decode(aes_json["key"])
        iv = base64.b64decode(aes_json["iv"])
        aad = base64.b64decode(aes_json["aad"])
        return key, iv, aad


# AES-GCM encryption method
def encrypt(key, plain_txt, associated_data, iv):
    # Construct an AES-GCM Cipher object with the given key and a
    # randomly generated IV.
    encryptor = Cipher(
        algorithms.AES(key), modes.GCM(iv), backend=default_backend()
    ).encryptor()

    # associated_data will be authenticated but not encrypted,
    # it must also be passed in on decryption.
    encryptor.authenticate_additional_data(associated_data)

    # Encrypt the plaintext and get the associated ciphertext.
    # GCM does not require padding.
    ciphertext = encryptor.update(plain_txt) + encryptor.finalize()

    return (ciphertext, encryptor.tag)


# AES-GCM decryption method
def decrypt(key, associated_data, iv, ciphertext, tag):
    # Construct a Cipher object, with the key, iv, and additionally the
    # GCM tag used for authenticating the message.
    decryptor = Cipher(
        algorithms.AES(key), modes.GCM(iv, bytes(tag)), backend=default_backend()
    ).decryptor()

    # We put associated_data back in or the tag will fail to verify
    # when we finalize the decryptor.
    decryptor.authenticate_additional_data(associated_data)

    # Decryption gets us the authenticated plaintext.
    # If the tag does not match an InvalidTag exception will be raised.
    return decryptor.update(ciphertext) + decryptor.finalize()


def split_cipher_txt_assembled(input):
    return input[0:-32], input[-32:-20], input[-20:-4], input[-4:]


def get_defaults_output_name(mode, input_file):
    suffix = "_enc"
    if mode == "dec":
        suffix = "_dec"
    start_pos = input_file.rfind("/")
    dot_pos = input_file.rfind(".")
    if dot_pos == -1:
        return input_file[start_pos + 1 :] + suffix
    else:
        return input_file[start_pos + 1 : dot_pos] + suffix + input_file[dot_pos:]


if __name__ == "__main__":
    args = get_args()
    with open(args.input_file, "rb") as f:
        input = f.read()
        input_bytes = bytearray(input)
        # print("Debug print: input: " + input_bytes.hex())

    user_key, user_iv, user_aad = get_aes_meta(args.aes_data_file)
    output = None
    if args.mode == "enc":
        cipher_txt, tag = encrypt(user_key, input_bytes, user_aad, user_iv)
        # print("Debug print: cipher txt: " + cipher_txt.hex() + " iv: " + user_iv.hex() + " tag: " + tag.hex() + " aad: " + user_aad.hex())
        cipher_txt_assembled = cipher_txt + user_iv + tag + user_aad
        if args.cipher_txt_encoding == "b64":
            output = base64.b64encode(cipher_txt_assembled)
        elif args.cipher_txt_encoding == "hex":
            output = base64.b16encode(cipher_txt_assembled)
        else:
            output = cipher_txt_assembled
        # print("Debug print: cipher txt(b64): " + output.hex())
    elif args.mode == "dec":
        if args.cipher_txt_encoding == "b64":
            cipher_txt_assembled = base64.b64decode(input_bytes)
        if args.cipher_txt_encoding == "hex":
            cipher_txt_assembled = base64.b16decode(input_bytes, casefold=True)
        else:
            cipher_txt_assembled = input_bytes
        cipher_txt, user_iv, tag, user_aad = split_cipher_txt_assembled(
            cipher_txt_assembled
        )
        print(
            "Debug print: cipher txt: "
            + cipher_txt.hex()
            + " iv: "
            + user_iv.hex()
            + " tag: "
            + tag.hex()
            + " aad: "
            + user_aad.hex()
        )
        output = decrypt(user_key, user_aad, user_iv, cipher_txt, tag)
        # print("Debug print: plain txt(b64): " + output.hex())
    elif args.mode == "nenc":
        if args.cipher_txt_encoding == "b64":
            output = base64.b64encode(input_bytes)
        elif args.cipher_txt_encoding == "hex":
            output = base64.b16encode(input_bytes)
        else:
            output = input_bytes
    else:
        assert False

    if not args.output_file:
        output_file = get_defaults_output_name(args.mode, args.input_file)
    else:
        output_file = args.output_file

    with open(output_file, "wb") as f:
        f.write(output)

# so far only tested with default values.
#   python aes_encrypt.py --mode=enc --input_file=test.txt --cipher_txt_encoding=b64 
#   python aes_encrypt.py --mode=dec --input_file=test_enc.txt --cipher_txt_encoding=b64
