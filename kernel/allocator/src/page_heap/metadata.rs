// Copyright (c) 2026 vivo Mobile Communication Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::HeapError;
use core::alloc::Layout;

// Low two bits are tags. Heads reserve six more bits for log2(alignment).
const TAG_MASK: usize = 0b11;
const LARGE_HEAD: usize = 0b10;
const LARGE_TAIL: usize = 0b11;
const ALIGN_BITS: u32 = 6;
const COUNT_SHIFT: u32 = 2 + ALIGN_BITS;
const ALIGN_MASK: usize = (1 << ALIGN_BITS) - 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PageOwner {
    words: [usize; 2],
}

impl PageOwner {
    pub const EMPTY: Self = Self { words: [0; 2] };

    pub(super) fn is_empty(&self) -> bool {
        self.words == [0; 2]
    }
    pub(super) fn can_encode_pages(pages: usize) -> bool {
        pages > 0 && pages <= usize::MAX >> COUNT_SHIFT
    }
    pub(super) fn can_encode_index(index: usize) -> bool {
        index <= usize::MAX >> 2
    }

    pub(super) fn large_head(pages: usize, layout: Layout) -> Result<Self, HeapError> {
        let align = layout.align().trailing_zeros();
        if !Self::can_encode_pages(pages) || align > ALIGN_MASK as u32 || layout.size() == 0 {
            return Err(HeapError::InvalidRequest);
        }
        Ok(Self {
            words: [
                (pages << COUNT_SHIFT) | ((align as usize) << 2) | LARGE_HEAD,
                layout.size(),
            ],
        })
    }

    pub(super) fn large_tail(first: usize) -> Result<Self, HeapError> {
        if !Self::can_encode_index(first) {
            return Err(HeapError::InvalidRequest);
        }
        Ok(Self {
            words: [(first << 2) | LARGE_TAIL, 0],
        })
    }

    pub(super) fn head(self) -> Result<(usize, Layout), HeapError> {
        let pages = self.words[0] >> COUNT_SHIFT;
        let exponent = (self.words[0] >> 2) & ALIGN_MASK;
        if self.words[0] & TAG_MASK != LARGE_HEAD || pages == 0 || self.words[1] == 0 {
            return Err(HeapError::CorruptMetadata);
        }
        let align = 1usize
            .checked_shl(exponent as u32)
            .ok_or(HeapError::CorruptMetadata)?;
        let layout = Layout::from_size_align(self.words[1], align)
            .map_err(|_| HeapError::CorruptMetadata)?;
        Ok((pages, layout))
    }

    pub(super) fn tail_index(self) -> Result<Option<usize>, HeapError> {
        if self.words[0] & TAG_MASK != LARGE_TAIL {
            return Ok(None);
        }
        if self.words[1] != 0 {
            return Err(HeapError::CorruptMetadata);
        }
        Ok(Some(self.words[0] >> 2))
    }

    pub(super) fn same_encoding(self, other: Self) -> bool {
        self.words == other.words
    }
}
