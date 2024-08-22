# KeyService

KeyService manages the decryption keys of the system users. A list of information stored in key service are listed below as key value pairs.

* system-user:
  * password table
* model user:
  * userId-MRENCLAVE : inputDecryptionKey
* model provider:
  * userId-modelName-keyInfoPostfix : modelDecryptionKey;
  * userId-modelName-userAccessInfoPostfix : a set of userIds;
  * userId-modelName-enclaveAccessInfoPostfix : a set of MRENCLAVE;

Currently, it is single threaded. It creates one WOLFSSL context inside the enclave and listen to the port. The logic inside would check decode the json request and execute the corresponding logic to the request type.
