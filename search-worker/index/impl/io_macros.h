/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAKES_SEARCHWORKER_INDEX_IMPL_IOMACRO_H_
#define HAKES_SEARCHWORKER_INDEX_IMPL_IOMACRO_H_

#include <cstdint>

/*************************************************************
 * I/O macros
 *
 * we use macros so that we have a line number to report in abort
 * (). This makes debugging a lot easier. The IOReader or IOWriter is
 * always called f and thus is not passed in as a macro parameter.
 **************************************************************/

#define READANDCHECK(ptr, n)                         \
    {                                                \
        uint64_t ret = (*f)(ptr, sizeof(*(ptr)), n);   \
        assert(ret == (n));                          \
    }

#define READ1(x) READANDCHECK(&(x), 1)

// will fail if we write 256G of data at once...
#define READVECTOR(vec)                                              \
    {                                                                \
        uint64_t size;                                                 \
        READANDCHECK(&size, 1);                                      \
        assert(size >= 0 && size < (uint64_t{1} << 40)); \
        (vec).resize(size);                                          \
        READANDCHECK((vec).data(), size);                            \
    }

#define READSTRING(s)                     \
    {                                     \
        uint64_t size = (s).size();         \
        WRITEANDCHECK(&size, 1);          \
        WRITEANDCHECK((s).c_str(), size); \
    }

#define WRITEANDCHECK(ptr, n)                         \
    {                                                 \
        uint64_t ret = (*f)(ptr, sizeof(*(ptr)), n);    \
        assert(ret == (n));                           \
    }

#define WRITE1(x) WRITEANDCHECK(&(x), 1)

#define WRITEVECTOR(vec)                   \
    {                                      \
        uint64_t size = (vec).size();        \
        WRITEANDCHECK(&size, 1);           \
        WRITEANDCHECK((vec).data(), size); \
    }

// read/write xb vector for backwards compatibility of IndexFlat

#define WRITEXBVECTOR(vec)                         \
    {                                              \
        assert((vec).size() % 4 == 0); \
        uint64_t size = (vec).size() / 4;            \
        WRITEANDCHECK(&size, 1);                   \
        WRITEANDCHECK((vec).data(), size * 4);     \
    }

#define READXBVECTOR(vec)                                            \
    {                                                                \
        uint64_t size;                                                 \
        READANDCHECK(&size, 1);                                      \
        assert(size >= 0 && size < (uint64_t{1} << 40)); \
        size *= 4;                                                   \
        (vec).resize(size);                                          \
        READANDCHECK((vec).data(), size);                            \
    }

#endif  // HAKES_SEARCHWORKER_INDEX_IMPL_IOMACRO_H_
