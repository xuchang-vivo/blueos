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

//! Builds page tables using architecture-provided descriptors and frame allocation.
//!
//! Inactive tables are reclaimed on drop, including partially built trees.
//! Call `retain_tables` before installing a root in hardware. Modifying active
//! tables requires a separate TLB protocol.

use super::{FrameAllocator, PageFormat};
use crate::mm::{
    kernel_phys_to_virt, MemAttribute, MemFlags, MemType, PhysicalAddress, VirtualAddress,
};
use core::{marker::PhantomData, mem::size_of};

#[must_use = "dropping an inactive mapper releases its page tables"]
pub struct PageMapper<F: FrameAllocator, P: PageFormat = crate::arch::page::PageEntry> {
    root_addr: usize,
    owns_tables: bool,
    allocator: F,
    format: PhantomData<P>,
}

impl<F: FrameAllocator, P: PageFormat> PageMapper<F, P> {
    /// Allocate and clear a new root table.
    ///
    /// # Safety
    /// P must describe the configured hardware format. Every frame returned by
    /// the allocator must be exclusively owned, writable through the kernel
    /// direct mapping, and remain valid for the lifetime of these page tables.
    pub unsafe fn create(allocator: F) -> Result<Self, &'static str> {
        if !P::PAGE_SIZE.is_power_of_two()
            || P::PAGE_SIZE < size_of::<usize>()
            || P::LEVELS == 0
            || P::PHYS_ADDR_BITS >= usize::BITS as u8
            || P::VIRT_ADDR_BITS >= usize::BITS as u8
        {
            return Err("unsupported page-table geometry");
        }
        let mut mapper = Self {
            root_addr: 0,
            owns_tables: false,
            allocator,
            format: PhantomData,
        };
        mapper.root_addr = mapper.allocate_table()?;
        mapper.owns_tables = true;
        Ok(mapper)
    }

    /// Keep the table tree permanently allocated, even when this mapper drops.
    /// Call before installing the root on any CPU. This does not manage mapped
    /// data pages; their owner must separately guarantee their lifetime.
    pub fn retain_tables(&mut self) {
        self.owns_tables = false;
    }

    pub fn root_address(&self) -> PhysicalAddress {
        PhysicalAddress::new(self.root_addr)
    }

    fn allocate_table(&mut self) -> Result<usize, &'static str> {
        let addr = self
            .allocator
            .alloc_frame()
            .ok_or("out of page-table frames")?
            .into_raw();
        if addr & (P::PAGE_SIZE - 1) != 0 || addr >= (1usize << P::PHYS_ADDR_BITS) {
            // SAFETY: the unpublished allocation has no users.
            unsafe { self.allocator.free_frame(PhysicalAddress::new(addr)) }
                .expect("failed to return an unusable page-table frame");
            return Err("invalid page-table frame");
        }
        let table = kernel_phys_to_virt(PhysicalAddress::new(addr)).into_raw() as *mut u8;
        // SAFETY: create's caller guarantees exclusive, writable backing frames.
        unsafe { table.write_bytes(0, P::PAGE_SIZE) };
        Ok(addr)
    }

    fn valid_va(addr: usize) -> bool {
        let mask = (1usize << P::VIRT_ADDR_BITS) - 1;
        addr <= mask || addr >= !mask
    }

    fn entry_ptr(table: usize, va: usize, level: u8) -> Result<*mut usize, &'static str> {
        let index = P::get_index(va, level);
        if index >= P::PAGE_SIZE / size_of::<usize>() {
            return Err("page-table index exceeds table size");
        }
        let table = kernel_phys_to_virt(PhysicalAddress::new(table)).into_raw() as *mut usize;
        // SAFETY: the index is in bounds of a page-table frame.
        Ok(unsafe { table.add(index) })
    }

    fn leaf_entry(&self, va: usize) -> Option<usize> {
        if !Self::valid_va(va) {
            return None;
        }
        let mut table = self.root_addr;
        for level in 0..P::LEVELS {
            let entry = Self::entry_ptr(table, va, level).ok()?;
            // SAFETY: tables remain live while this mapper is borrowed.
            let entry = unsafe { entry.read() };
            if level == P::LEVELS - 1 {
                return P::is_valid(entry).then_some(entry);
            }
            table = P::table_address(entry)?;
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn leaf_descriptor(&self, addr: usize) -> Option<usize> {
        self.leaf_entry(addr)
    }

    /// Translate a mapped address, including its offset within the page.
    pub fn translate(&self, addr: usize) -> Option<PhysicalAddress> {
        let frame = P::table_address(self.leaf_entry(addr)?)?;
        Some(PhysicalAddress::new(frame + (addr & (P::PAGE_SIZE - 1))))
    }

    fn leaf_slot(&mut self, va: usize) -> Result<*mut usize, &'static str> {
        let mut table = self.root_addr;
        for level in 0..P::LEVELS - 1 {
            let slot = Self::entry_ptr(table, va, level)?;
            // SAFETY: this mapper exclusively builds inactive tables.
            let entry = unsafe { slot.read() };
            table = if !P::is_valid(entry) {
                let next = self.allocate_table()?;
                unsafe { slot.write(P::table_descriptor(next)) };
                next
            } else {
                P::table_address(entry).ok_or("mapping conflicts with a block descriptor")?
            };
        }
        Self::entry_ptr(table, va, P::LEVELS - 1)
    }

    /// Allocate the intermediate tables for a virtual range before assigning
    /// backing pages. Existing leaf entries are left unchanged. This lets a
    /// caller consume the remaining frames without starving table allocation.
    pub fn prepare_range(&mut self, virt: VirtualAddress, len: usize) -> Result<(), &'static str> {
        if len == 0 {
            return Ok(());
        }
        if (virt.into_raw() | len) & (P::PAGE_SIZE - 1) != 0 {
            return Err("mapping range must be page aligned");
        }
        let end = virt
            .into_raw()
            .checked_add(len)
            .ok_or("virtual mapping range overflow")?;
        if !Self::valid_va(virt.into_raw())
            || !Self::valid_va(end - 1)
            || (virt.into_raw() >> P::VIRT_ADDR_BITS) != ((end - 1) >> P::VIRT_ADDR_BITS)
        {
            return Err("mapping range exceeds address space");
        }
        let span = P::PAGE_SIZE
            .checked_mul(P::PAGE_SIZE / size_of::<usize>())
            .ok_or("page-table span overflow")?;
        let mut addr = virt.into_raw();
        while addr < end {
            self.leaf_slot(addr)?;
            addr += (span - (addr & (span - 1))).min(end - addr);
        }
        Ok(())
    }

    /// Map an aligned range in an inactive table. Existing mappings are rejected.
    /// On allocation failure, some of the range may already have been mapped;
    /// the caller must discard the table rather than install a partial layout.
    pub fn map_range(
        &mut self,
        virt: VirtualAddress,
        phys: PhysicalAddress,
        len: usize,
        attributes: MemAttribute,
    ) -> Result<(), &'static str> {
        if attributes
            .flags
            .contains(MemFlags::WRITE | MemFlags::EXECUTE)
        {
            return Err("writable executable mappings are not supported");
        }
        if len == 0 {
            return Ok(());
        }
        if (virt.into_raw() | phys.into_raw() | len) & (P::PAGE_SIZE - 1) != 0 {
            return Err("mapping range must be page aligned");
        }
        let end = virt
            .into_raw()
            .checked_add(len)
            .ok_or("virtual mapping range overflow")?;
        let phys_end = phys
            .into_raw()
            .checked_add(len)
            .ok_or("physical mapping range overflow")?;
        if !Self::valid_va(virt.into_raw())
            || !Self::valid_va(end - 1)
            || (virt.into_raw() >> P::VIRT_ADDR_BITS) != ((end - 1) >> P::VIRT_ADDR_BITS)
            || phys_end > (1usize << P::PHYS_ADDR_BITS)
        {
            return Err("mapping range exceeds address space");
        }
        // FIXME: Temporarily allow writable executable mappings so the ELF loader tests pass.
        // Restore W^X enforcement once the loader uses dedicated executable mappings.
        if !attributes.flags.contains(MemFlags::READ)
            || (matches!(attributes.memory_type, MemType::Device)
                && attributes.flags.contains(MemFlags::EXECUTE))
        {
            return Err("mapping must be readable and device memory cannot be executable");
        }
        for addr in (virt.into_raw()..end).step_by(P::PAGE_SIZE) {
            if self.leaf_entry(addr).is_some() {
                return Err("virtual page is already mapped");
            }
        }
        for offset in (0..len).step_by(P::PAGE_SIZE) {
            let slot = self.leaf_slot(virt.into_raw() + offset)?;
            unsafe {
                slot.write(P::get_translate(
                    phys.into_raw() + offset,
                    attributes,
                    false,
                ))
            };
        }
        Ok(())
    }
}

impl<F: FrameAllocator, P: PageFormat> Drop for PageMapper<F, P> {
    fn drop(&mut self) {
        if self.owns_tables {
            // SAFETY: these tables have not been published to hardware. Only
            // child tables are released; leaf backing belongs to the caller.
            unsafe { self.release_table(self.root_addr, 0) };
        }
    }
}

impl<F: FrameAllocator, P: PageFormat> PageMapper<F, P> {
    unsafe fn release_table(&mut self, table: usize, level: u8) {
        if level + 1 < P::LEVELS {
            let entries =
                kernel_phys_to_virt(PhysicalAddress::new(table)).into_raw() as *const usize;
            for index in 0..P::PAGE_SIZE / size_of::<usize>() {
                if let Some(child) = P::table_address(unsafe { entries.add(index).read() }) {
                    unsafe { self.release_table(child, level + 1) };
                }
            }
        }
        unsafe { self.allocator.free_frame(PhysicalAddress::new(table)) }
            .expect("invalid page-table frame during cleanup");
    }
}
