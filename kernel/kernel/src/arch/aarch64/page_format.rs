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

use crate::mm::paging::{PageLevel, PageFormat};

pub static A64_4K_39: PageFormat = PageFormat {
    va_bits: 39,
    max_pa_bits: 48,
    levels: &[
        // root, L1
        PageLevel {
            index_shift: 30,
            index_bits: 9,
            table_bytes: 4096,
            table_align: 4096,
            leaf_allowed: true, // 1 GiB block
        },
        // L2
        PageLevel {
            index_shift: 21,
            index_bits: 9,
            table_bytes: 4096,
            table_align: 4096,
            leaf_allowed: true, // 2 MiB block
        },
        // L3
        PageLevel {
            index_shift: 12,
            index_bits: 9,
            table_bytes: 4096,
            table_align: 4096,
            leaf_allowed: true, // 4 KiB page
        },
    ],
};

pub static A64_4K_48: PageFormat = PageFormat {
    va_bits: 48,
    max_pa_bits: 48,
    levels: &[
        // L0 root; block descriptors are not allowed at this level.
        PageLevel {
            index_shift: 39,
            index_bits: 9,
            table_bytes: 4096,
            table_align: 4096,
            leaf_allowed: false,
        },
        // L1
        PageLevel {
            index_shift: 30,
            index_bits: 9,
            table_bytes: 4096,
            table_align: 4096,
            leaf_allowed: true, // 1 GiB block
        },
        // L2
        PageLevel {
            index_shift: 21,
            index_bits: 9,
            table_bytes: 4096,
            table_align: 4096,
            leaf_allowed: true, // 2 MiB block
        },
        // L3
        PageLevel {
            index_shift: 12,
            index_bits: 9,
            table_bytes: 4096,
            table_align: 4096,
            leaf_allowed: true, // 4 KiB page
        },
    ],
};

