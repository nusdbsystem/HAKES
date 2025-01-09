# HAKES python packages

Python packages of HAKES

## Index module

The index module supports building and training an index on a vector dataset. `build_index.py` is a convenient tool for that purpose.

```sh
python build_index.py --N 1000000 --d 1024 --dr 256 --nlist 2048 --data_path <data-path> --output_path <output-dir-for-index>
```

It accepts the dataset as binary of float32 values and output the `findex.bin` and `uindex.bin` to be loaded by `searchworker`.
