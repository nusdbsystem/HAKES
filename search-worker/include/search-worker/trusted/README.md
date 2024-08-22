# README

For SGX node, intrinsic headers and openmp header need to be copied over from the local gcc installation. This is because they are not bundled with linux-sgx distribution. For example,

```sh
cp /usr/lib/gcc/x86_64-linux-gnu/9/include/omp.h include/search-worker/trusted/
cp /usr/lib/gcc/x86_64-linux-gnu/9/include/*intrin.h include/search-worker/trusted/intrinsics/
cp /usr/lib/gcc/x86_64-linux-gnu/9/include/mm_malloc.h include/search-worker/trusted/intrinsics/
```

After copying over the headers this folder should have the following structure

```txt
.
├── Enclave.edl
├── intrinsics
│   ├── adxintrin.h
│   ├── ammintrin.h
│   ├── avx2intrin.h
│   ├── avx5124fmapsintrin.h
│   ├── avx5124vnniwintrin.h
│   ├── avx512bitalgintrin.h
│   ├── avx512bwintrin.h
│   ├── avx512cdintrin.h
│   ├── avx512dqintrin.h
│   ├── avx512erintrin.h
│   ├── avx512fintrin.h
│   ├── avx512ifmaintrin.h
│   ├── avx512ifmavlintrin.h
│   ├── avx512pfintrin.h
│   ├── avx512vbmi2intrin.h
│   ├── avx512vbmi2vlintrin.h
│   ├── avx512vbmiintrin.h
│   ├── avx512vbmivlintrin.h
│   ├── avx512vlbwintrin.h
│   ├── avx512vldqintrin.h
│   ├── avx512vlintrin.h
│   ├── avx512vnniintrin.h
│   ├── avx512vnnivlintrin.h
│   ├── avx512vpopcntdqintrin.h
│   ├── avx512vpopcntdqvlintrin.h
│   ├── avxintrin.h
│   ├── bmi2intrin.h
│   ├── bmiintrin.h
│   ├── bmmintrin.h
│   ├── cetintrin.h
│   ├── cldemoteintrin.h
│   ├── clflushoptintrin.h
│   ├── clwbintrin.h
│   ├── clzerointrin.h
│   ├── emmintrin.h
│   ├── f16cintrin.h
│   ├── fma4intrin.h
│   ├── fmaintrin.h
│   ├── fxsrintrin.h
│   ├── gfniintrin.h
│   ├── ia32intrin.h
│   ├── immintrin.h
│   ├── lwpintrin.h
│   ├── lzcntintrin.h
│   ├── mmintrin.h
│   ├── mm_malloc.h
│   ├── movdirintrin.h
│   ├── mwaitxintrin.h
│   ├── nmmintrin.h
│   ├── pconfigintrin.h
│   ├── pkuintrin.h
│   ├── pmmintrin.h
│   ├── popcntintrin.h
│   ├── prfchwintrin.h
│   ├── rdseedintrin.h
│   ├── rtmintrin.h
│   ├── sgxintrin.h
│   ├── shaintrin.h
│   ├── smmintrin.h
│   ├── tbmintrin.h
│   ├── tmmintrin.h
│   ├── vaesintrin.h
│   ├── vpclmulqdqintrin.h
│   ├── waitpkgintrin.h
│   ├── wbnoinvdintrin.h
│   ├── wmmintrin.h
│   ├── x86intrin.h
│   ├── xmmintrin.h
│   ├── xopintrin.h
│   ├── xsavecintrin.h
│   ├── xsaveintrin.h
│   ├── xsaveoptintrin.h
│   ├── xsavesintrin.h
│   └── xtestintrin.h
├── omp.h
└── README.md
```
