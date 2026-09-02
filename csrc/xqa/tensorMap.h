#pragma once

#include <cuda.h>

uint32_t getElemBytes(CUtensorMapDataType_enum dataType);

CUtensorMap makeTensorMapForContiguousKVCache(void const* addr, CUtensorMapDataType_enum dataType,
                                              uint32_t headElems, uint32_t nbKHeads,
                                              uint32_t maxCacheLen, uint32_t beamWidth,
                                              uint32_t batchSize, uint32_t partElems,
                                              uint32_t nbTokens);

CUtensorMap makeTensorMapForPagedKVCache(void const* addr, CUtensorMapDataType_enum dataType,
                                         uint32_t headElems, uint32_t nbKHeads,
                                         uint32_t tokensPerPage, uint32_t partElems,
                                         uint32_t nbTokensPerTile, uint64_t stride_page,
                                         uint64_t stride_token, uint64_t stride_head);

// Byte-typed, unswizzled map over a paged tensor of compressed rows
// ([page][token][head][rowBytes] with the given byte strides).  A box covers
// one page of one head: tokensPerPage rows of partBytes each.
CUtensorMap makeTensorMapForPackedKVPages(void const* addr, uint32_t rowBytes, uint32_t nbKHeads,
                                          uint32_t tokensPerPage, uint32_t partBytes,
                                          uint64_t stride_page_bytes, uint64_t stride_token_bytes,
                                          uint64_t stride_head_bytes);
