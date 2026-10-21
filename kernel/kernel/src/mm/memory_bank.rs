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

//! Abstraction over the memory banks available on the hardware platform.
//!
//! A memory bank is a contiguous range of physical memory described by the
//! platform (e.g. via a devicetree or firmware-provided memory map). This
//! module collects those banks so the rest of the memory management
//! subsystem can enumerate usable physical memory without knowing how a
//! particular board reports it.

use crate::mm::PhysicalAddress;

pub struct MemoryBank {
    /// The start of the memory bank, inclusive.
    pub start: PhysicalAddress,
    /// The end of the memory bank, exclusive.
    pub end: PhysicalAddress,
}
