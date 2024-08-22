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

"""
  Generate key service request set for user and owner
  (encoding of cipher text is base64)
"""

import argparse
from aes_encrypt import get_aes_meta, encrypt
import json
import hashlib
import base64


def get_args():
    parser = argparse.ArgumentParser(
        description="generate user/owner key service request set"
    )
    parser.add_argument("--mode", required=True, choices={"user", "owner"})
    parser.add_argument("--aes_data_file", help="json file of metadata to use in aes")
    parser.add_argument(
        "--req_ctx_file", required=True, help="the source to gen request sets"
    )
    parser.add_argument("--output_dir", help="output directory")
    return parser.parse_args()


def gen_id(id_key):
    return hashlib.sha256(id_key).hexdigest()


def gen_req_json(id_key, aad, iv, type, payload):
    payload_bytes = json.dumps(payload).encode()
    print(payload_bytes)
    cipher_txt, tag = encrypt(id_key, payload_bytes, aad, iv)
    print(cipher_txt.hex())
    cipher_txt_assembled = cipher_txt + iv + tag + aad
    return {
        "type": type,
        "user_id": gen_id(id_key),
        "payload": base64.b64encode(cipher_txt_assembled).decode(),
    }


def gen_add_request_key_req(id_key, aad, iv, model_id, enclave_id, decrypt_key):
    payload = {
        "model_id": model_id,
        "mrenclave": enclave_id,
        "decrypt_key": decrypt_key,
    }
    return gen_req_json(id_key, aad, iv, 1, payload)


def gen_up_model_key_req(id_key, aad, iv, model_id, decrypt_key):
    payload = {"model_id": model_id, "decrypt_key": decrypt_key}
    return gen_req_json(id_key, aad, iv, 2, payload)


def gen_grant_model_access_req(id_key, aad, iv, model_id, enclave_id, user_id):
    payload = {"model_id": model_id, "mrenclave": enclave_id, "user_id": user_id}
    return gen_req_json(id_key, aad, iv, 3, payload)


def gen_user_request_set(id_key, aad, iv, src_json, output_dir):
    if "enclave_id" in src_json and "decrypt_key" in src_json:
        with open(output_dir + "/add_request_key_req.json", "w") as f:
            json.dump(
                gen_add_request_key_req(
                    id_key,
                    aad,
                    iv,
                    src_json["model_id"],
                    src_json["enclave_id"],
                    src_json["decrypt_key"],
                ),
                f,
            )
    else:
        print("user src file in invalid format")


def gen_owner_request_set(id_key, aad, iv, src_json, output_dir):
    if (
        "model_id" in src_json
        and "decrypt_key" in src_json
        and "enclave_id" in src_json
        and "user_id" in src_json
    ):
        with open(output_dir + "/up_model_key_req.json", "w") as f:
            json.dump(
                gen_up_model_key_req(
                    id_key, aad, iv, src_json["model_id"], src_json["decrypt_key"]
                ),
                f,
            )
        with open(output_dir + "/grant_model_access_req.json", "w") as f:
            json.dump(
                gen_grant_model_access_req(
                    id_key,
                    aad,
                    iv,
                    src_json["model_id"],
                    src_json["enclave_id"],
                    src_json["user_id"],
                ),
                f,
            )
    else:
        print("owner src file in invalid format")


if __name__ == "__main__":
    args = get_args()
    user_key, user_iv, user_aad = get_aes_meta(args.aes_data_file)

    # req_ctx_file is a required field
    with open(args.req_ctx_file, "rb") as f:
        src_json = json.load(f)
        if args.mode == "user":
            gen_user_request_set(user_key, user_aad, user_iv, src_json, args.output_dir)
        else:
            # args.mode == 'owner'
            gen_owner_request_set(
                user_key, user_aad, user_iv, src_json, args.output_dir
            )
