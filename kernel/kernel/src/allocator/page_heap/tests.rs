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
    source::{FirstFitSource, PageSource, PendingPages},
    PageHeap,
};
use crate::{mm::paging::FrameStats, sync::SpinLock};
use alloc::sync::{Arc, Weak};
use allocator_crate::page_heap::{PageOwner, PageRequest, PageSpan, PageWindow, PAGE_SIZE};
use blueos_test_macro::test;
use core::{
    alloc::Layout,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct CountingSource {
    inner: FirstFitSource,
    fail_next: AtomicBool,
    live_pages: AtomicUsize,
    probe: SpinLock<Option<Weak<PageHeap<CountingSource>>>>,
}
impl CountingSource {
    fn new() -> Self {
        Self {
            inner: FirstFitSource,
            fail_next: AtomicBool::new(false),
            live_pages: AtomicUsize::new(0),
            probe: SpinLock::new(None),
        }
    }
    fn assert_unlocked(&self) {
        let heap = self.probe.irqsave_lock().as_ref().and_then(Weak::upgrade);
        if let Some(heap) = heap {
            let guard = heap
                .state
                .try_lock()
                .expect("PageSource called under heap lock");
            drop(guard);
        }
    }
}
// SAFETY: forwards exclusive ownership to/from the real frame source.
unsafe impl PageSource for CountingSource {
    fn allocate(&self, request: PageRequest) -> Option<PageSpan> {
        self.assert_unlocked();
        if self.fail_next.swap(false, Ordering::AcqRel) {
            return None;
        }
        let span = self.inner.allocate(request)?;
        self.live_pages.fetch_add(span.pages(), Ordering::AcqRel);
        Some(span)
    }
    unsafe fn release(&self, span: PageSpan) {
        self.assert_unlocked();
        unsafe {
            self.inner.release(span);
        }
        self.live_pages.fetch_sub(span.pages(), Ordering::AcqRel);
    }
    fn frame_stats(&self) -> FrameStats {
        self.inner.frame_stats()
    }
    fn largest_free_run_bytes(&self) -> usize {
        self.inner.largest_free_run_bytes()
    }
}

#[must_use]
struct TestMetadata {
    ptr: NonNull<u8>,
    layout: Layout,
    pages: usize,
}
impl TestMetadata {
    fn new(pages: usize) -> Self {
        let layout = Layout::array::<PageOwner>(pages).unwrap();
        let ptr = NonNull::new(unsafe { alloc::alloc::alloc_zeroed(layout) }).unwrap();
        Self { ptr, layout, pages }
    }
}
impl Drop for TestMetadata {
    fn drop(&mut self) {
        unsafe { alloc::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

#[must_use]
struct KernelFixture {
    heap: Arc<PageHeap<CountingSource>>,
    metadata: TestMetadata,
}
impl KernelFixture {
    fn new() -> Self {
        let pages = crate::boards::PHYS_DRAM_SIZE as usize / PAGE_SIZE;
        let metadata = TestMetadata::new(pages);
        let base = crate::mm::kernel_phys_to_virt(crate::mm::PhysicalAddress::new(
            crate::boards::PHYS_DRAM_BASE as usize,
        ))
        .into_raw();
        let window =
            unsafe { PageWindow::new(NonNull::new(base as *mut u8).unwrap(), pages) }.unwrap();
        let heap = Arc::new(PageHeap::with_source(CountingSource::new()));
        *heap.source.probe.irqsave_lock() = Some(Arc::downgrade(&heap));
        // Fixture owns stable raw metadata storage; heap drops first, no Box retag.
        let owners: &'static mut [PageOwner] = unsafe {
            core::slice::from_raw_parts_mut(metadata.ptr.as_ptr().cast(), metadata.pages)
        };
        unsafe {
            heap.init(window, owners).unwrap();
        }
        Self { heap, metadata }
    }
}
impl Drop for KernelFixture {
    fn drop(&mut self) {
        assert_eq!(Arc::strong_count(&self.heap), 1);
        assert_eq!(self.heap.source.live_pages.load(Ordering::Acquire), 0);
        assert_eq!(
            self.metadata.pages,
            crate::boards::PHYS_DRAM_SIZE as usize / PAGE_SIZE
        );
    }
}

// Catches leaked OOM candidates, missing final-page return, wrong alignment,
// and taking the physical source lock while holding the heap lock.
#[test]
fn private_heap_rolls_back_oom_and_returns_pages_outside_lock() {
    let f = KernelFixture::new();
    assert_eq!(f.heap.heap_stats().backing_bytes, 0);
    f.heap.source.fail_next.store(true, Ordering::Release);
    assert!(f
        .heap
        .alloc(Layout::from_size_align(24, 8).unwrap())
        .is_none());
    assert_eq!(f.heap.source.live_pages.load(Ordering::Acquire), 0);
    let small = f
        .heap
        .alloc(Layout::from_size_align(24, 8).unwrap())
        .unwrap();
    let large = f
        .heap
        .alloc(Layout::from_size_align(8193, 65536).unwrap())
        .unwrap();
    assert_eq!(large.as_ptr().addr() & (65536 - 1), 0);
    unsafe {
        f.heap.deallocate_unknown_align(small.as_ptr());
        f.heap.deallocate_unknown_align(large.as_ptr());
    }
    assert_eq!(f.heap.source.live_pages.load(Ordering::Acquire), 0);
    assert_eq!(f.heap.heap_stats().backing_bytes, 0);
}

// Catches premature page release on commit and missing rollback on drop.
#[test]
fn pending_pages_roll_back_or_transfer_ownership() {
    let f = KernelFixture::new();
    let request = PageRequest {
        pages: 1,
        align: PAGE_SIZE,
    };
    let pending = PendingPages::acquire(&f.heap.source, request).unwrap();
    assert_eq!(pending.span().pages(), 1);
    assert_eq!(f.heap.source.live_pages.load(Ordering::Acquire), 1);
    drop(pending);
    assert_eq!(f.heap.source.live_pages.load(Ordering::Acquire), 0);
    let pending = PendingPages::acquire(&f.heap.source, request).unwrap();
    let span = {
        let _guard = f.heap.state.irqsave_lock();
        pending.commit()
    };
    assert_eq!(f.heap.source.live_pages.load(Ordering::Acquire), 1);
    unsafe {
        f.heap.source.release(span);
    }
}

// Catches losing existing slab slots on OOM and incorrect layout-aware release.
#[test]
fn private_heap_uses_existing_slots_without_requesting_pages() {
    let f = KernelFixture::new();
    let layout = Layout::from_size_align(24, 8).unwrap();
    let first = f.heap.alloc(layout).unwrap();
    f.heap.source.fail_next.store(true, Ordering::Release);
    let second = f.heap.alloc(layout).unwrap();
    assert_ne!(first, second);
    assert_eq!(f.heap.source.live_pages.load(Ordering::Acquire), 1);
    assert_eq!(f.heap.heap_stats().live_objects, 2);
    assert!(f.heap.source.fail_next.load(Ordering::Acquire));
    unsafe {
        f.heap.dealloc(first.as_ptr(), layout);
        f.heap.dealloc(second.as_ptr(), layout);
    }
    assert_eq!(f.heap.heap_stats().backing_bytes, 0);
}

#[test]
fn c_requests_and_failed_realloc_preserve_old_allocation() {
    let f = KernelFixture::new();
    assert!(f.heap.malloc(0).is_null());
    assert!(f.heap.malloc_align(17, 3).is_null());
    assert!(f.heap.calloc(usize::MAX, 2).is_null());
    let old = f.heap.malloc_align(63, 64);
    assert!(!old.is_null());
    unsafe {
        old.write_bytes(0x5a, 63);
    }
    f.heap.source.fail_next.store(true, Ordering::Release);
    let result = unsafe { f.heap.c_realloc(old, 8193) };
    assert!(result.is_null());
    for i in 0..63 {
        assert_eq!(unsafe { old.add(i).read() }, 0x5a);
    }
    assert_eq!(f.heap.heap_stats().live_objects, 1);
    let zeroed = f.heap.calloc(17, 19);
    assert!(!zeroed.is_null());
    for i in 0..17 * 19 {
        assert_eq!(unsafe { zeroed.add(i).read() }, 0);
    }
    unsafe {
        f.heap.free(old);
        f.heap.free(zeroed);
    }
    assert_eq!(f.heap.source.live_pages.load(Ordering::Acquire), 0);
}

#[test]
fn realloc_moves_preserving_data_and_alignment() {
    let f = KernelFixture::new();
    let layout = Layout::from_size_align(63, 64).unwrap();
    let old = f.heap.alloc(layout).unwrap();
    unsafe {
        old.as_ptr().write_bytes(0x37, 63);
    }
    let new = unsafe { f.heap.realloc(old.as_ptr(), layout, 8193) }.unwrap();
    assert_eq!(new.as_ptr().addr() % 64, 0);
    for i in 0..63 {
        assert_eq!(unsafe { new.as_ptr().add(i).read() }, 0x37);
    }
    assert!(f.heap.size_of_allocation(new) >= 8193);
    unsafe {
        f.heap
            .dealloc(new.as_ptr(), Layout::from_size_align(8193, 64).unwrap());
    }
}

#[test]
fn c_realloc_null_zero_and_large_alignment() {
    let f = KernelFixture::new();
    let p = unsafe { f.heap.c_realloc(core::ptr::null_mut(), 37) };
    assert!(!p.is_null());
    assert!(unsafe { f.heap.c_realloc(p, 0) }.is_null());
    let p = f.heap.malloc_align(8193, 65536);
    assert!(!p.is_null());
    unsafe {
        p.write_bytes(0x4c, 8193);
    }
    let q = unsafe { f.heap.c_realloc(p, 20000) };
    assert!(!q.is_null());
    assert_eq!(q.addr() % 65536, 0);
    for i in 0..8193 {
        assert_eq!(unsafe { q.add(i).read() }, 0x4c);
    }
    unsafe {
        f.heap.free_align(q, 65536);
        f.heap.free(core::ptr::null_mut());
    }
}

// A smaller slab layout must remain valid for a later Rust deallocation.
#[test]
fn realloc_shrinks_slab_and_accepts_new_layout() {
    let f = KernelFixture::new();
    let old = Layout::from_size_align(63, 8).unwrap();
    let ptr = f.heap.alloc(old).unwrap();
    unsafe {
        ptr.as_ptr().write_bytes(0x27, 12);
    }
    let resized = unsafe { f.heap.realloc(ptr.as_ptr(), old, 12) }.unwrap();
    assert_eq!(resized, ptr);
    for i in 0..12 {
        assert_eq!(unsafe { resized.as_ptr().add(i).read() }, 0x27);
    }
    unsafe {
        f.heap
            .dealloc(resized.as_ptr(), Layout::from_size_align(12, 8).unwrap());
    }
    assert_eq!(f.heap.source.live_pages.load(Ordering::Acquire), 0);
}
