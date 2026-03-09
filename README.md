# HAKES

HAKES is an embedding vector data search system. It features modular and disaggregated architecture designs across the three data management modules, data storage, vector search, and embedding model hosting. It aims for resource efficiency, scalability, and fine-grained scaling in cloud/clustered deployment. Moreover, HAKES provides a proof-of-concept (PoC) implementation of security-protection mode leveraging Intel Software Guard Extensions (SGX) to operate in the untrusted environment.

## VLDB 2025

To reproduce our experiments in our VLDB 2025 paper, please consider the [HAKES-Search Repo](https://github.com/nusdbsystem/HAKES-Search), a cleaned codebase we used for paper submission. We will also release the instructions for experiment data preparation and the trained index parameters there. 

## Pause for SGX support

With a shifted focus for developing more features around vector search, we stopped maintaining code compatibility for SGX deployment. Please use the `sgx-support` branch for the original HAKES support of SGX.

## Key modules

* `hakes-worker`: exposes Key-value and AKNN search interface.
* `embed-worker`: host embedding models. It supports the tflm and TVM C runtime to run model inference on CPU.
* `embed-endpoint`: allow connection to external embedding services. We provide plugins for the OpenAI embedding service and HuggingFace inference endpoints.
* `fnpacker`: middleware when embed-workers are deployed as functions on a serverless platform (Current implementation demonstrates usage with Apache OpenWhisk). It can expose an HTTP endpoint with one or more function endpoint backends.
* `search-worker`: serves a two-phase vector search: a fast filter phase with quantized index, followed by an accurate refine phase with full vectors. It allows injecting fine-tuned index parameters online, which enables adaptation for specific query workloads.
* `hakes-store`: an efficient fault-tolerant storage layer designed for shared storage architecture. It uses the LSM-tree to organize data and boost resource efficiency for cloud deployment with cloud shared storage and serverless computing.

For Intel SGX security protection mode.

* requires SGX-enabled Linux servers and attestation service set up over the servers according to the documentation on Intel SGX Data Center Attestation Primitives.
* `hakes-worker`, `embed-worker`, `search-worker` can be compiled with SGX support to perform data processing on plain-text data only inside a trusted execution environment (enclave) set up by SGX.
* `key-service`: stores secret keys for data encryption and manages access control for the enclaves.

## Deployment

All components of HAKES are containerised, and instructions to build the images can be found under `docker`.

## Ongoing development

* A CLI tool to facilitate management of HAKES deployments for multiple datasets
* Additional documentation and guides
* Examples

## Reference

Please cite our publication when you use HAKES in your research or development.

* Guoyu Hu, Shaofeng Cai, Tien Tuan Anh Dinh, Zhongle Xie, Cong Yue, Gang Chen, and Beng Chin Ooi. HAKES: Scalable Vector Database for Embedding Search Service. PVLDB, 18(9): 3049 - 3062, 2025. doi:10.14778/3746405.3746427

## Contact

Beng Chin Ooi (ooibc@zju.edu.cn)

Zhongle Xie   (xiezl@zju.edu.cn)

Cong Yue      (yuecong@comp.nus.edu.sg)
