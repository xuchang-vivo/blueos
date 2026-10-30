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

//! The kernel's shared virtual address space and permanent page tables.

use super::{
    address_space::Vma,
    kernel_phys_to_virt,
    memory_bank::MemoryBank,
    paging::{first_fit::FirstFitFrameAllocator, mapper::PageMapper, PageFormat},
    CachePolicy, MemAttribute, MemFlags, MemType, PhysicalAddress,
};
use crate::arch::page::PageEntry;
use spin::Once;

// Eight RAM segments and at most one device region per board bank.
const REGION_COUNT: usize = 8 + crate::boards::MEMORY_BANKS.len();

static KERNEL_SPACE: Once<KernelAddressSpace> = Once::new();

#[must_use]
pub struct KernelAddressSpace {
    regions: [Option<Vma>; REGION_COUNT],
    mapper: PageMapper<&'static FirstFitFrameAllocator, PageEntry>,
}

impl KernelAddressSpace {
    pub fn get() -> Option<&'static Self> {
        KERNEL_SPACE.get()
    }

    pub fn root_address(&self) -> PhysicalAddress {
        self.mapper.root_address()
    }

    #[cfg(test)]
    pub(crate) fn leaf_descriptor(&self, addr: usize) -> Option<usize> {
        self.find_region(addr)?;
        self.mapper.leaf_descriptor(addr)
    }

    pub fn find_region(&self, addr: usize) -> Option<Vma> {
        self.regions
            .iter()
            .flatten()
            .find(|region| region.contains(addr))
            .copied()
    }

    pub fn translate(&self, addr: usize) -> Option<PhysicalAddress> {
        // A root-table walk alone cannot distinguish the TTBR0 and TTBR1
        // halves. Only addresses in the kernel's recorded regions are valid.
        self.find_region(addr)?;
        self.mapper.translate(addr)
    }

    /// Map a physical memory region to the kernel's virtual address space.
    /// The physical address will be added a KERNEL_OFFSET
    fn map_linear(
        &mut self,
        start: PhysicalAddress,
        end: PhysicalAddress,
        flags: MemFlags,
        memory_type: MemType,
    ) -> Result<(), &'static str> {
        if start == end {
            return Ok(());
        }
        let len = end
            .into_raw()
            .checked_sub(start.into_raw())
            .ok_or("reversed kernel mapping")?;
        let virt = kernel_phys_to_virt(start);
        let virt_end = virt
            .into_raw()
            .checked_add(len)
            .ok_or("kernel mapping overflow")?;
        self.mapper.map_range(
            virt,
            start,
            len,
            MemAttribute {
                flags: flags | MemFlags::GLOBAL,
                memory_type,
            },
        )?;
        let vma_flags = MemFlags::READ | flags;
        self.record_region(virt.into_raw(), virt_end, vma_flags)
    }

    fn record_region(
        &mut self,
        start: usize,
        end: usize,
        flags: MemFlags,
    ) -> Result<(), &'static str> {
        if self
            .regions
            .iter()
            .flatten()
            .any(|region| start < region.end() && region.start() < end)
        {
            return Err("overlapping kernel address-space regions");
        }
        let region = Vma::new(start, end, flags).map_err(|_| "invalid kernel region")?;
        let slot = self
            .regions
            .iter_mut()
            .find(|region| region.is_none())
            .ok_or("kernel region table is full")?;
        *slot = Some(region);
        Ok(())
    }

    fn bootstrap() -> Result<Self, &'static str> {
        use super::{kernel_virt_to_phys, VirtualAddress};
        use core::ptr::addr_of;

        unsafe extern "C" {
            static __text_start: u8;
            static __rodata_start: u8;
            static __data_start: u8;
            static __bss_start: u8;
            static __sys_stack_start: u8;
            static __sys_stack_end: u8;
            static _end: u8;
        }
        let physical = |addr: usize| kernel_virt_to_phys(VirtualAddress::new(addr)).into_raw();
        let ram_start = crate::boards::PHYS_DRAM_BASE as usize;
        let ram_end = ram_start
            .checked_add(crate::boards::PHYS_DRAM_SIZE as usize)
            .ok_or("physical RAM range overflow")?;
        // The linker aligns each boundary to a page. The final boot-table page
        // is covered too; heap pages use the writable, non-executable direct map.
        let boundaries = [
            ram_start,
            physical(addr_of!(__text_start) as usize),
            physical(addr_of!(__rodata_start) as usize),
            physical(addr_of!(__data_start) as usize),
            physical(addr_of!(__bss_start) as usize),
            physical(addr_of!(__sys_stack_start) as usize),
            physical(addr_of!(__sys_stack_end) as usize),
            physical(addr_of!(_end) as usize),
            ram_end,
        ];
        if boundaries.windows(2).any(|range| range[0] > range[1]) {
            return Err("kernel sections do not fit in physical RAM");
        }
        if boundaries
            .iter()
            .any(|addr| addr & (PageEntry::PAGE_SIZE - 1) != 0)
        {
            return Err("kernel section boundaries must be page aligned");
        }
        let allocator = FirstFitFrameAllocator::instance();
        // SAFETY: boot initialized the physical allocator and the boot direct
        // map. New table frames are disjoint from the reserved kernel image,
        // and these tables are retained permanently by KERNEL_SPACE.
        let mapper = unsafe { PageMapper::create(allocator)? };
        let mut space = Self {
            regions: [None; REGION_COUNT],
            mapper,
        };
        let normal = MemType::Normal(CachePolicy::WriteBack);
        for (index, range) in boundaries.windows(2).enumerate() {
            let flags = match index {
                1 => MemFlags::READ | MemFlags::EXECUTE,
                2 => MemFlags::READ,
                _ => MemFlags::READ | MemFlags::WRITE,
            };
            space.map_linear(
                PhysicalAddress::new(range[0]),
                PhysicalAddress::new(range[1]),
                flags,
                normal,
            )?;
        }
        for bank in crate::boards::MEMORY_BANKS {
            let MemoryBank::MMIO((start, end)) = bank else {
                continue;
            };
            space.map_linear(
                *start,
                *end,
                MemFlags::READ | MemFlags::WRITE,
                MemType::Device,
            )?;
        }
        // SAFETY: these tables cover the kernel and direct-mapped RAM. Static
        // region metadata and page tables were constructed without heap use.
        space.mapper.retain_tables();
        unsafe { crate::arch::mmu::activate_kernel_table(space.root_address()) };
        Ok(space)
    }
}

pub(crate) fn init_kernel_space() -> Result<(), &'static str> {
    KERNEL_SPACE.try_call_once(KernelAddressSpace::bootstrap)?;
    Ok(())
}
