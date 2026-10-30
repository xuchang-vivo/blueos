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

//! The kernel shared virtual address space and permanent page tables.

use super::{
    address_space::Vma,
    kernel_phys_to_virt,
    memory_bank::{MemoryBank, MemoryRegion},
    paging::{first_fit::FirstFitFrameAllocator, mapper::PageMapper, PageFormat},
    CachePolicy, MemAttribute, MemFlags, MemType, PhysicalAddress,
};
use crate::arch::page::PageEntry;
use heapless::Vec;
use spin::Once;

const PAGE_SIZE: usize = <PageEntry as PageFormat>::PAGE_SIZE;

// Up to six image sections plus one boot-data region per RAM bank, the free
// segments carved out of every RAM bank, and one region per board bank.
const REGION_COUNT: usize = 12 + 4 * crate::boards::MEMORY_BANKS.len();

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
        // halves. Only addresses in the kernel recorded regions are valid.
        self.find_region(addr)?;
        self.mapper.translate(addr)
    }

    /// Map a physical memory region to the kernel virtual address space.
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

    fn bootstrap(regions: &[MemoryRegion]) -> Result<Self, &'static str> {
        let mut reserved: Vec<&MemoryRegion, 16> = Vec::new();
        for region in regions {
            if region.is_reserved {
                reserved
                    .push(region)
                    .map_err(|_| "too many reserved regions")?;
            }
        }
        for region in &reserved {
            let end = region
                .base
                .into_raw()
                .checked_add(region.size)
                .ok_or("reserved region overflow")?;
            if region.base.into_raw() % PAGE_SIZE != 0 || end % PAGE_SIZE != 0 {
                return Err("reserved region boundaries must be page aligned");
            }
            let inside = regions.iter().any(|ram| {
                !ram.is_reserved
                    && region.base.into_raw() >= ram.base.into_raw()
                    && end <= ram.base.into_raw().saturating_add(ram.size)
            });
            if region.size > 0 && !inside {
                return Err("reserved region lies outside every RAM bank");
            }
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
        // Reserved regions take precedence: they are mapped with their own
        // flags before the remaining RAM is handed to the linear map.
        for region in &reserved {
            space.map_linear(
                region.base,
                PhysicalAddress::new(region.base.into_raw() + region.size),
                region.flags,
                normal,
            )?;
        }
        for ram in regions.iter().filter(|region| !region.is_reserved) {
            for (base, size) in carve_ram_holes(ram, reserved.iter().copied())? {
                space.map_linear(
                    base,
                    PhysicalAddress::new(base.into_raw() + size),
                    MemFlags::READ | MemFlags::WRITE,
                    normal,
                )?;
            }
        }
        for bank in crate::boards::MEMORY_BANKS {
            let MemoryBank::MMIO((start, end)) = bank else {
                continue;
            };
            space.map_linear(*start, *end, MemFlags::READ | MemFlags::WRITE, MemType::Device)?;
        }
        // SAFETY: these tables cover the kernel and direct-mapped RAM. Static
        // region metadata and page tables were constructed without heap use.
        space.mapper.retain_tables();
        unsafe { crate::arch::mmu::activate_kernel_table(space.root_address()) };
        Ok(space)
    }
}

/// Split `ram` into the page-aligned sub-ranges not covered by `reserved`.
///
/// Reservations are clipped to the RAM window; every surviving boundary must
/// stay page aligned so `map_range` accepts the carved segments. The returned
/// segments are sorted and never overlap a reservation.
pub(crate) fn carve_ram_holes<'a>(
    ram: &MemoryRegion,
    reserved: impl IntoIterator<Item = &'a MemoryRegion>,
) -> Result<Vec<(PhysicalAddress, usize), 8>, &'static str> {
    let ram_start = ram.base.into_raw();
    let ram_end = ram_start.checked_add(ram.size).ok_or("RAM region overflow")?;
    if ram.size == 0 {
        return Ok(Vec::new());
    }
    if ram_start % PAGE_SIZE != 0 || ram_end % PAGE_SIZE != 0 {
        return Err("RAM region boundaries must be page aligned");
    }
    let mut spans: Vec<(usize, usize), 8> = Vec::new();
    for region in reserved {
        let start = region.base.into_raw().max(ram_start);
        let end = region
            .base
            .into_raw()
            .checked_add(region.size)
            .ok_or("reserved region overflow")?
            .min(ram_end);
        if start >= end {
            continue;
        }
        if start % PAGE_SIZE != 0 || end % PAGE_SIZE != 0 {
            return Err("reservation boundaries must be page aligned");
        }
        spans
            .push((start, end))
            .map_err(|_| "too many reservations in one RAM bank")?;
    }
    spans.sort_unstable();
    let mut holes: Vec<(PhysicalAddress, usize), 8> = Vec::new();
    let mut cursor = ram_start;
    for (start, end) in spans {
        if start > cursor {
            holes
                .push((PhysicalAddress::new(cursor), start - cursor))
                .map_err(|_| "too many free segments in one RAM bank")?;
        }
        cursor = cursor.max(end);
    }
    if cursor < ram_end {
        holes
            .push((PhysicalAddress::new(cursor), ram_end - cursor))
            .map_err(|_| "too many free segments in one RAM bank")?;
    }
    Ok(holes)
}

pub(crate) fn init_kernel_space(regions: &[MemoryRegion]) -> Result<(), &'static str> {
    KERNEL_SPACE.try_call_once(|| KernelAddressSpace::bootstrap(regions))?;
    Ok(())
}
