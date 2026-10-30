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

use allocator_crate::page_heap::{PageRequest, PageSpan, PAGE_SIZE};
use core::ptr::NonNull;

use crate::mm::{
    kernel_phys_to_virt, kernel_virt_to_phys,
    paging::{first_fit::FirstFitFrameAllocator, FrameStats},
    VirtualAddress,
};

/// Transfers exclusive ownership of contiguous, writable direct-mapped pages.
///
/// # Safety
/// Successful allocations must be disjoint, aligned to the request and contain
/// at least the requested pages. A single-page request must return exactly one
/// page. Pages remain valid until released exactly once to this same source.
/// Failed allocation must not consume pages. Implementations must be safe for
/// concurrent callers and must not allocate metadata through the object heap.
pub unsafe trait PageSource: Sync {
    fn allocate(&self, request: PageRequest) -> Option<PageSpan>;

    /// # Safety
    /// The caller must exclusively own this entire live span from this source,
    /// with no remaining accesses, and relinquish ownership exactly once.
    unsafe fn release(&self, span: PageSpan);
    fn frame_stats(&self) -> FrameStats;
    fn largest_free_run_bytes(&self) -> usize;
}

pub struct FirstFitSource;

// SAFETY: FirstFit marks disjoint, aligned runs allocated before exposing them.
// Owned frames retain rollback until the direct-map span has been validated.
unsafe impl PageSource for FirstFitSource {
    fn allocate(&self, request: PageRequest) -> Option<PageSpan> {
        if request.pages == 0 || request.align < PAGE_SIZE || !request.align.is_power_of_two() {
            return None;
        }
        let frames = FirstFitFrameAllocator::instance()
            .alloc_owned_frames_aligned(request.pages, request.align / PAGE_SIZE)?;
        let start = NonNull::new(kernel_phys_to_virt(frames.address()).into_raw() as *mut u8)?;
        if start.as_ptr().addr() & (request.align - 1) != 0 {
            return None;
        }
        // SAFETY: frames owns a contiguous writable run through the direct map.
        let span = unsafe { PageSpan::new(start, frames.count()) }.ok()?;
        let _address = frames.into_raw();
        Some(span)
    }

    unsafe fn release(&self, span: PageSpan) {
        let physical = kernel_virt_to_phys(VirtualAddress::new(span.start().as_ptr().addr()));
        // SAFETY: the caller transfers this entire exclusive live allocation.
        unsafe { FirstFitFrameAllocator::instance().free_frames(physical, span.pages()) }
            .expect("invalid page source release");
    }

    fn frame_stats(&self) -> FrameStats {
        FirstFitFrameAllocator::instance()
            .frame_stats()
            .expect("page source is not initialized")
    }

    fn largest_free_run_bytes(&self) -> usize {
        FirstFitFrameAllocator::instance()
            .largest_free_run()
            .expect("page source is not initialized")
            * PAGE_SIZE
    }
}

/// Exclusive, unpublished pages; dropping this handle returns them to source.
#[must_use = "uncommitted pages are returned to their source on drop"]
pub struct PendingPages<'a, P: PageSource> {
    source: &'a P,
    span: Option<PageSpan>,
}

impl<'a, P: PageSource> PendingPages<'a, P> {
    pub fn acquire(source: &'a P, request: PageRequest) -> Option<Self> {
        Some(Self {
            source,
            span: Some(source.allocate(request)?),
        })
    }

    pub fn span(&self) -> PageSpan {
        self.span.expect("pending pages already committed")
    }

    /// Transfer ownership without executing Drop, even inside the heap lock.
    pub fn commit(mut self) -> PageSpan {
        let span = self.span.take().expect("pending pages already committed");
        // Only a borrowed source remains; forgetting it leaks no resource.
        core::mem::forget(self);
        span
    }
}

impl<P: PageSource> Drop for PendingPages<'_, P> {
    fn drop(&mut self) {
        if let Some(span) = self.span.take() {
            // SAFETY: this handle exclusively owns its unpublished allocation.
            unsafe { self.source.release(span) };
        }
    }
}
