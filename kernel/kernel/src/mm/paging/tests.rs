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

use super::{
    first_fit::{init_frame_states, FirstFitFrameAllocator, FrameState},
    mapper::PageMapper,
    FrameDeallocationError, FrameStats, PageFormat,
};
use crate::mm::{
    kernel_phys_to_virt, kernel_virt_to_phys, memory_bank::MemoryRegion, CachePolicy, MemAttribute,
    MemFlags, MemType, PhysicalAddress, VirtualAddress,
};
use alloc::{
    alloc::{alloc, dealloc, Layout},
    boxed::Box,
};
use blueos_test_macro::test;
use core::ptr::NonNull;

const PAGE_SIZE: usize = 4096;

/// Private suballocator over exclusively owned, physically contiguous heap
/// backing. Its state and backing outlive every allocation used by each test.
#[must_use]
struct Fixture<const N: usize> {
    allocator: FirstFitFrameAllocator,
    _states: Box<[FrameState; N]>,
    backing: NonNull<u8>,
    layout: Layout,
    managed_base: PhysicalAddress,
}

impl<const N: usize> Fixture<N> {
    fn new(states: [FrameState; N]) -> Self {
        let layout = Layout::from_size_align(N * PAGE_SIZE, PAGE_SIZE).unwrap();
        let backing = NonNull::new(unsafe { alloc(layout) }).expect("fixture backing allocation");
        let mut states = Box::new(states);
        let allocator = FirstFitFrameAllocator::new();
        let base = kernel_virt_to_phys(VirtualAddress::new(backing.as_ptr() as usize));
        // SAFETY: the fixture owns stable backing and metadata allocations. All
        // returned frames are released before fixture teardown, and metadata
        // outlives the allocator field.
        unsafe {
            allocator
                .init(
                    base,
                    PAGE_SIZE,
                    &mut *(&mut *states as *mut [FrameState; N]),
                )
                .unwrap();
        }
        Self {
            allocator,
            _states: states,
            backing,
            layout,
            managed_base: base,
        }
    }

    fn with_misaligned_base(states: [FrameState; N]) -> Self {
        let layout = Layout::from_size_align((N + 1) * PAGE_SIZE, PAGE_SIZE).unwrap();
        let backing = NonNull::new(unsafe { alloc(layout) }).unwrap();
        let physical = kernel_virt_to_phys(VirtualAddress::new(backing.as_ptr() as usize));
        let skip = usize::from((physical.into_raw() & (4 * PAGE_SIZE - 1)) == 0);
        let managed_base = PhysicalAddress::new(physical.into_raw() + skip * PAGE_SIZE);
        let mut states = Box::new(states);
        let allocator = FirstFitFrameAllocator::new();
        // SAFETY: the N-page managed window lies inside this fixture's N+1
        // backing pages, with the same lifetime guarantees as `new`.
        unsafe {
            allocator
                .init(
                    managed_base,
                    PAGE_SIZE,
                    &mut *(&mut *states as *mut [FrameState; N]),
                )
                .unwrap();
        }
        Self {
            allocator,
            _states: states,
            backing,
            layout,
            managed_base,
        }
    }

    fn base(&self) -> PhysicalAddress {
        self.managed_base
    }

    fn frame(&self, index: usize) -> PhysicalAddress {
        PhysicalAddress::new(self.base().into_raw() + index * PAGE_SIZE)
    }
}

impl<const N: usize> Drop for Fixture<N> {
    fn drop(&mut self) {
        // All guards borrow this fixture and have already been dropped.
        unsafe { dealloc(self.backing.as_ptr(), self.layout) };
    }
}

#[test]
fn aligned_runs_use_absolute_physical_alignment() {
    let fixture = Fixture::with_misaligned_base([FrameState::Free; 8]);
    let allocator = &fixture.allocator;
    assert_ne!(fixture.base().into_raw() % (4 * PAGE_SIZE), 0);
    let run = allocator.alloc_owned_frames_aligned(2, 4).unwrap();
    assert_eq!(run.address().into_raw() % (4 * PAGE_SIZE), 0);
    let stats = allocator.frame_stats().unwrap();
    assert_eq!(
        (stats.allocated, stats.free, stats.peak_allocated),
        (2, 6, 2)
    );
    drop(run);
    let after = allocator.frame_stats().unwrap();
    assert_eq!(
        (after.allocated, after.free, after.peak_allocated),
        (0, 8, 2)
    );
}

#[test]
fn aligned_search_preserves_skipped_frames_on_success_and_failure() {
    let fixture = Fixture::with_misaligned_base([FrameState::Free; 8]);
    let allocator = &fixture.allocator;
    let before = allocator.frame_stats().unwrap();
    assert!(allocator.alloc_frames_aligned(8, 4).is_none());
    assert_eq!(allocator.frame_stats().unwrap(), before);
    let run = allocator.alloc_owned_frames_aligned(2, 4).unwrap();
    let first = allocator.alloc_owned_frames(1).unwrap();
    assert_eq!(first.address(), fixture.base());
    drop(first);
    drop(run);
    assert_eq!(allocator.frame_stats().unwrap().free, 8);
}

#[test]
fn invalid_alignment_does_not_consume_frames() {
    let fixture = Fixture::new([FrameState::Free; 8]);
    let allocator = &fixture.allocator;
    let before = allocator.frame_stats().unwrap();
    assert!(allocator.alloc_frames_aligned(1, 0).is_none());
    assert!(allocator.alloc_frames_aligned(1, 3).is_none());
    assert!(allocator.alloc_frames_aligned(usize::MAX, 1).is_none());
    assert!(allocator.alloc_frames_aligned(0, 1).is_none());
    assert!(allocator
        .alloc_frames_aligned(1, 1usize << (usize::BITS - 1))
        .is_none());
    assert_eq!(allocator.frame_stats().unwrap(), before);
    assert_eq!(
        allocator.alloc_owned_frames(1).unwrap().address(),
        fixture.base()
    );
}

#[test]
fn reserved_holes_bound_free_runs_and_preserve_statistics() {
    use FrameState::{Free as F, Reserved as R};
    let fixture = Fixture::new([F, R, F, F, R, F]);
    let allocator = &fixture.allocator;
    let initial = FrameStats {
        total: 6,
        reserved: 2,
        allocated: 0,
        free: 4,
        peak_allocated: 0,
    };
    assert_eq!(allocator.frame_stats().unwrap(), initial);
    assert_eq!(allocator.usage().unwrap().used(), 2);
    assert_eq!(allocator.largest_free_run(), Some(2));
    assert!(allocator.alloc_frames_aligned(3, 1).is_none());
    assert_eq!(allocator.frame_stats().unwrap(), initial);
    let run = allocator.alloc_owned_frames_aligned(2, 1).unwrap();
    assert_eq!(run.address(), fixture.frame(2));
    assert_eq!(allocator.largest_free_run(), Some(1));
    assert_eq!(allocator.usage().unwrap().used(), 4);
    let allocated = FrameStats {
        allocated: 2,
        free: 2,
        peak_allocated: 2,
        ..initial
    };
    assert_eq!(allocator.frame_stats().unwrap(), allocated);
    assert_eq!(
        unsafe { allocator.free_frames(run.address(), 3) },
        Err(FrameDeallocationError::Reserved)
    );
    assert_eq!(allocator.frame_stats().unwrap(), allocated);
    let first = allocator.alloc_owned_frames(1).unwrap();
    assert_eq!(first.address(), fixture.base());
    drop(first);
    drop(run);
    assert_eq!(allocator.largest_free_run(), Some(2));
    assert_eq!(
        allocator.frame_stats().unwrap(),
        FrameStats {
            peak_allocated: 3,
            ..initial
        }
    );
}

#[test]
fn frame_statistics_track_initial_allocations_and_empty_free_runs() {
    let fresh = FirstFitFrameAllocator::new();
    assert_eq!(fresh.frame_stats(), None);
    assert_eq!(fresh.largest_free_run(), None);
    use FrameState::{Allocated as A, Reserved as R};
    let fixture = Fixture::new([R, A]);
    let allocator = &fixture.allocator;
    assert_eq!(
        allocator.frame_stats(),
        Some(FrameStats {
            total: 2,
            reserved: 1,
            allocated: 1,
            free: 0,
            peak_allocated: 1,
        })
    );
    assert_eq!(allocator.largest_free_run(), Some(0));
    unsafe { allocator.free_frame(fixture.frame(1)).unwrap() };
    assert_eq!(
        allocator.frame_stats(),
        Some(FrameStats {
            total: 2,
            reserved: 1,
            allocated: 0,
            free: 1,
            peak_allocated: 1,
        })
    );
    assert_eq!(allocator.largest_free_run(), Some(1));
}

#[test]
fn frame_regions_reserve_partial_pages_and_holes() {
    let base = 0x10000;
    let region = |start, size, reserved| MemoryRegion {
        base: PhysicalAddress::new(start),
        size,
        is_reserved: reserved,
        flags: MemFlags::READ | MemFlags::WRITE,
        name: "test",
    };
    // Put reservations before RAM to verify precedence does not depend on order.
    let regions = [
        region(base + 2 * PAGE_SIZE + 1, PAGE_SIZE, true),
        region(base + 17, 6 * PAGE_SIZE - 28, false),
        region(base + 7 * PAGE_SIZE, PAGE_SIZE, false),
    ];
    let mut states = [FrameState::Free; 8];
    init_frame_states(PhysicalAddress::new(base), PAGE_SIZE, &mut states, &regions).unwrap();
    use FrameState::{Free as F, Reserved as R};
    assert_eq!(states, [R, F, R, R, F, R, R, F]);
}

#[test]
fn frame_allocate_free_reuse_and_exhaustion() {
    use FrameState::{Free as F, Reserved as R};
    let fixture = Fixture::new([R, F, F, R, F]);
    let allocator = &fixture.allocator;
    assert_eq!(allocator.usage().unwrap().free(), 3);
    let first = allocator.alloc_frames(2).unwrap();
    assert_eq!(first, fixture.frame(1));
    let last = allocator.alloc_frame().unwrap();
    assert_eq!(last, fixture.frame(4));
    assert!(allocator.alloc_frame().is_none());
    assert_eq!(allocator.usage().unwrap().free(), 0);
    unsafe {
        allocator.free_frames(first, 2).unwrap();
    }
    assert_eq!(allocator.alloc_frame().unwrap(), first);
    unsafe {
        allocator.free_frame(first).unwrap();
        allocator.free_frame(last).unwrap();
    }
    assert_eq!(allocator.usage().unwrap().free(), 3);
}

#[test]
fn frame_contiguous_failure_preserves_fragmented_pages() {
    use FrameState::{Free as F, Reserved as R};
    let fixture = Fixture::new([F, R, F, R, F, F]);
    let allocator = &fixture.allocator;
    assert!(allocator.alloc_frames(3).is_none());
    assert!(allocator.alloc_frames(0).is_none());
    assert!(allocator.alloc_frames(usize::MAX).is_none());
    assert_eq!(allocator.usage().unwrap().free(), 4);
    let run = allocator.alloc_owned_frames(2).unwrap();
    assert_eq!(run.address(), fixture.frame(4));
    let first = allocator.alloc_owned_frames(1).unwrap();
    assert_eq!(first.address(), fixture.frame(0));
    drop(run);
    drop(first);
    assert_eq!(allocator.usage().unwrap().free(), 4);
}

#[test]
fn frame_free_rejects_invalid_reserved_and_duplicate_ranges() {
    use FrameDeallocationError::{InvalidRange, NotAllocated, NotInitialized, Reserved};
    use FrameState::{Free as F, Reserved as R};
    let fresh = FirstFitFrameAllocator::new();
    assert_eq!(
        unsafe { fresh.free_frame(PhysicalAddress::new(0)) },
        Err(NotInitialized)
    );
    let fixture = Fixture::new([F, F, R, F]);
    let allocator = &fixture.allocator;
    let run = allocator.alloc_frames(2).unwrap();
    unsafe {
        for (address, count) in [
            (PhysicalAddress::new(run.into_raw() - 1), 1),
            (PhysicalAddress::new(run.into_raw() + 1), 1),
            (fixture.frame(4), 1),
            (run, 0),
            (run, usize::MAX),
        ] {
            assert_eq!(allocator.free_frames(address, count), Err(InvalidRange));
        }
        assert_eq!(allocator.free_frames(run, 3), Err(Reserved));
        assert_eq!(allocator.usage().unwrap().free(), 1);
        allocator.free_frame(fixture.frame(1)).unwrap();
        assert_eq!(allocator.free_frames(run, 2), Err(NotAllocated));
        assert_eq!(allocator.usage().unwrap().free(), 2);
        allocator.free_frame(run).unwrap();
        assert_eq!(allocator.free_frame(run), Err(NotAllocated));
    }
    assert_eq!(allocator.usage().unwrap().free(), 3);
}

#[test]
fn frame_guards_rollback_and_zero_recycled_user_pages() {
    let fixture = Fixture::new([FrameState::Free; 3]);
    let allocator = &fixture.allocator;
    fn fail(allocator: &FirstFitFrameAllocator) -> Option<()> {
        let _first = allocator.alloc_owned_frames(2)?;
        let _second = allocator.alloc_owned_frames(2)?;
        Some(())
    }
    assert!(fail(allocator).is_none());
    assert_eq!(allocator.usage().unwrap().free(), 3);
    let frame = allocator.alloc_owned_frames(1).unwrap();
    let address = frame.address();
    let bytes = kernel_phys_to_virt(address).into_raw() as *mut u8;
    unsafe {
        bytes.write_bytes(0xa5, PAGE_SIZE);
    }
    drop(frame);
    let zeroed = allocator.alloc_zeroed_frame().unwrap();
    assert_eq!(zeroed.address(), address);
    assert!(unsafe { core::slice::from_raw_parts(bytes, PAGE_SIZE) }
        .iter()
        .all(|b| *b == 0));
    let raw = zeroed.into_raw();
    assert_eq!(allocator.usage().unwrap().free(), 2);
    unsafe {
        allocator.free_frame(raw).unwrap();
    }
    assert_eq!(allocator.usage().unwrap().free(), 3);
}

#[test]
fn page_table_allocation_failure_returns_all_table_frames() {
    let fixture = Fixture::new([FrameState::Free; 2]);
    let allocator = &fixture.allocator;
    for _ in 0..32 {
        let mut mapper = unsafe { PageMapper::<_>::create(allocator) }.unwrap();
        assert!(mapper
            .prepare_range(VirtualAddress::new(0x4000), PAGE_SIZE)
            .is_err());
        assert_eq!(allocator.usage().unwrap().free(), 0);
    }
    assert_eq!(allocator.usage().unwrap().free(), 2);
    {
        let _all = allocator.alloc_owned_frames(2).unwrap();
        assert!(unsafe { PageMapper::<_>::create(allocator) }.is_err());
    }
    assert_eq!(allocator.usage().unwrap().free(), 2);
}

#[test]
fn page_table_cleanup_keeps_leaf_backing_owned_by_caller() {
    let fixture = Fixture::new([FrameState::Free; 8]);
    let allocator = &fixture.allocator;
    let data = allocator.alloc_zeroed_frame().unwrap();
    let data_addr = data.address();
    {
        let mut mapper = unsafe { PageMapper::<_>::create(allocator) }.unwrap();
        mapper
            .map_range(
                VirtualAddress::new(0x8000),
                data_addr,
                PAGE_SIZE,
                MemAttribute {
                    flags: MemFlags::READ | MemFlags::WRITE | MemFlags::USER,
                    memory_type: MemType::Normal(CachePolicy::WriteBack),
                },
            )
            .unwrap();
        assert_eq!(mapper.translate(0x8000), Some(data_addr));
        assert_eq!(
            allocator.usage().unwrap().free(),
            8 - 1 - crate::arch::page::PageEntry::LEVELS as usize
        );
        // A separate branch shares the same data page. Cleanup must walk both
        // branches and must not release either leaf's backing allocation.
        mapper
            .map_range(
                VirtualAddress::new(0x4000_0000),
                data_addr,
                PAGE_SIZE,
                MemAttribute {
                    flags: MemFlags::READ | MemFlags::USER,
                    memory_type: MemType::Normal(CachePolicy::WriteBack),
                },
            )
            .unwrap();
        assert_eq!(mapper.translate(0x4000_0000), Some(data_addr));
        assert_eq!(
            allocator.usage().unwrap().free(),
            8 - 3 - crate::arch::page::PageEntry::LEVELS as usize
        );
    }
    assert_eq!(allocator.usage().unwrap().free(), 7);
    drop(data);
    assert_eq!(allocator.usage().unwrap().free(), 8);
}

#[test]
fn boot_runtime_frames_do_not_overlap_live_heap_buffers() {
    let buffer = alloc::vec![0xa5u8; 2 * PAGE_SIZE];
    let start = kernel_virt_to_phys(VirtualAddress::new(buffer.as_ptr() as usize)).into_raw();
    let page = FirstFitFrameAllocator::instance()
        .alloc_zeroed_frame()
        .unwrap();
    assert!(page.address().into_raw() < start || page.address().into_raw() >= start + buffer.len());
    assert!(buffer.iter().all(|byte| *byte == 0xa5));
}

#[cfg(target_arch = "aarch64")]
#[test]
fn heap_backing_uses_privileged_non_executable_leaf_entries() {
    let buffer = alloc::vec![0u8; PAGE_SIZE];
    let space = crate::mm::kernel_space::KernelAddressSpace::get().unwrap();
    let entry = space.leaf_descriptor(buffer.as_ptr() as usize).unwrap();
    assert_ne!(entry & (1usize << 53), 0); // AArch64 PXN
    assert_ne!(entry & (1usize << 54), 0); // AArch64 UXN
    assert_eq!((entry >> 6) & 0b11, 0); // AP: privileged read/write
    assert_eq!(entry & (1usize << 11), 0); // nG: global kernel mapping
}

#[test]
fn writable_executable_mapping_is_rejected_without_allocating_tables() {
    let fixture = Fixture::new([FrameState::Free; 8]);
    let mut mapper = unsafe { PageMapper::<_>::create(&fixture.allocator) }.unwrap();
    let before = fixture.allocator.frame_stats().unwrap();
    assert!(mapper
        .map_range(
            VirtualAddress::new(0x8000),
            fixture.base(),
            PAGE_SIZE,
            MemAttribute {
                flags: MemFlags::READ | MemFlags::WRITE | MemFlags::EXECUTE,
                memory_type: MemType::Normal(CachePolicy::WriteBack),
            },
        )
        .is_err());
    assert_eq!(mapper.translate(0x8000), None);
    assert_eq!(fixture.allocator.frame_stats().unwrap(), before);
}

#[test]
fn retained_kernel_table_is_not_reclaimed_on_mapper_drop() {
    let fixture = Fixture::new([FrameState::Free; 2]);
    let allocator = &fixture.allocator;
    let mut mapper = unsafe { PageMapper::<_>::create(allocator) }.unwrap();
    let root = mapper.root_address();
    mapper.retain_tables();
    drop(mapper);
    assert_eq!(allocator.usage().unwrap().free(), 1);
    // No hardware used this test root; explicitly return its transferred owner.
    unsafe {
        allocator.free_frame(root).unwrap();
    }
    assert_eq!(allocator.usage().unwrap().free(), 2);
}

#[test]
fn frame_concurrent_allocations_remain_exclusive_across_yields() {
    use crate::{scheduler, types::Arc};
    use core::sync::atomic::{AtomicUsize, Ordering};
    let done = Arc::new(AtomicUsize::new(0));
    for worker in 0..4 {
        let done = done.clone();
        crate::thread::spawn(move || {
            let allocator = FirstFitFrameAllocator::instance();
            for iteration in 0..32 {
                let run = allocator.alloc_owned_frames(2).unwrap();
                let bytes = kernel_phys_to_virt(run.address()).into_raw() as *mut u8;
                let pattern = (worker * 32 + iteration) as u8;
                unsafe {
                    bytes.write_bytes(pattern, 2 * PAGE_SIZE);
                }
                scheduler::yield_me();
                let contents = unsafe { core::slice::from_raw_parts(bytes, 2 * PAGE_SIZE) };
                assert!(contents.iter().all(|b| *b == pattern));
            }
            done.fetch_add(1, Ordering::Release);
        })
        .expect("spawn frame allocator test worker");
    }
    while done.load(Ordering::Acquire) != 4 {
        scheduler::yield_me();
    }
}

#[test]
fn carve_ram_holes_excludes_reserved_and_aligns() {
    use crate::mm::kernel_space::carve_ram_holes;
    let region = |base: usize, size: usize, reserved: bool| MemoryRegion {
        base: PhysicalAddress::new(base),
        size,
        is_reserved: reserved,
        flags: MemFlags::READ | MemFlags::WRITE,
        name: "test",
    };
    let ram = region(0x1000_0000, 0x4_0000, false);
    let reserved = [
        region(0x1001_0000, 0x1_0000, true),
        region(0x1002_0000, 0x1_0000, true),
    ];
    let holes = carve_ram_holes(&ram, reserved.iter()).unwrap();
    assert_eq!(
        holes.as_slice(),
        [
            (PhysicalAddress::new(0x1000_0000), 0x1_0000),
            (PhysicalAddress::new(0x1003_0000), 0x1_0000),
        ]
    );
}

#[test]
fn carve_ram_holes_rejects_non_page_aligned() {
    use crate::mm::kernel_space::carve_ram_holes;
    let region = |base: usize, size: usize, reserved: bool| MemoryRegion {
        base: PhysicalAddress::new(base),
        size,
        is_reserved: reserved,
        flags: MemFlags::READ | MemFlags::WRITE,
        name: "test",
    };
    let ram = region(0x2000_0000, 4 * PAGE_SIZE, false);
    let misaligned_base = [region(0x2000_1001, 2 * PAGE_SIZE, true)];
    assert!(carve_ram_holes(&ram, misaligned_base.iter()).is_err());
    let misaligned_end = [region(0x2000_2000, PAGE_SIZE + 1, true)];
    assert!(carve_ram_holes(&ram, misaligned_end.iter()).is_err());
    let aligned = [region(0x2000_2000, PAGE_SIZE, true)];
    assert!(carve_ram_holes(&ram, aligned.iter()).is_ok());
}
