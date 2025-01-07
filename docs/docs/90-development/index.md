# Development

## Building from source

For local development, it is more convenient to build and test directly from source code.

### Building in an Anaconda environment

To set up dependencies (e.g., cURL, Intel MKL) without root permission, you may create an [Anaconda](https://www.anaconda.com/download/success) environment and install prebuilt packages.

```bash
conda create -n hakes-dev
conda activate hakes-dev
conda install 'libcurl-static=7.87' mkl clang
```

Then, add the path to the installed C libraries to the compiler library search path.

```bash
conda env config vars set CPATH=${CPATH}:${CONDA_PREFIX}/include
conda env config vars set LIBRARY_PATH=${LIBRARY_PATH}:${CONDA_PREFIX}/lib
conda env config vars set LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
conda deactivate && conda activate hakes-dev
```

Clone and prepare code dependencies.

```bash
make preparation server_deps
```

Finally, go to the module directory (e.g., `hakes-worker/`) and build.

```bash
cd hakes-worker
make no_sgx -j
```
