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

pub struct PageLevel {
    pub index_shift: u8,
    pub index_bits: u8,
    pub table_bytes: usize,
    pub table_align: usize,
    pub leaf_allowed: bool,
}

pub struct PageFormat {
    pub va_bits: u8,
    pub max_pa_bits: u8,
    pub levels: &'static [PageLevel],
}

impl PageFormat {
    const fn level_count(&self) -> usize {
        self.levels.len()
    }

    // Safty: the validation of the va and pa is not checked here, 
    //        it should be checked before calling this function.
    pub fn index(&self, va: u64, level: usize) -> usize {
        let spec = &self.levels[level];
        let mask = (1u64 << spec.index_bits) - 1;
        ((va >> spec.index_shift) & mask) as usize
    }
}
