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

pub mod source;
#[cfg(test)]
mod tests;

use allocator_crate::page_heap::{
    classify, AllocationPlan, HeapError, HeapStats, PageHeap as RawHeap, PageOwner, PageRequest,
    PageWindow, ReleaseAction, SlabCommit, PAGE_SIZE,
};
use core::{alloc::Layout, ptr::NonNull};

use self::source::{FirstFitSource, PageSource, PendingPages};
use crate::sync::SpinLock;

pub struct PageHeap<P: PageSource = FirstFitSource> {
    state: SpinLock<Option<RawHeap<'static>>>,
    source: P,
}

impl PageHeap<FirstFitSource> {
    pub const fn new() -> Self {
        Self::with_source(FirstFitSource)
    }
}

impl<P: PageSource> PageHeap<P> {
    pub const fn with_source(source: P) -> Self {
        Self {
            state: SpinLock::new(None),
            source,
        }
    }

    /// Initialize once with a stable, exclusively owned page ownership table.
    ///
    /// # Safety
    /// `owners` must remain exclusively borrowed by this heap and must not
    /// overlap pages returned by the source. The window must cover the source's
    /// direct-mapped pages, which remain writable until returned to the source.
    pub unsafe fn init(
        &self,
        window: PageWindow,
        owners: &'static mut [PageOwner],
    ) -> Result<(), HeapError> {
        // Clear the ownership table before taking the heap lock.
        let raw = unsafe { RawHeap::new(window, owners) }?;
        let mut state = self.state.irqsave_lock();
        if state.is_some() {
            return Err(HeapError::Conflict);
        }
        *state = Some(raw);
        Ok(())
    }

    pub fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        match classify(layout).ok()? {
            AllocationPlan::Slab(class) => {
                {
                    let mut guard = self.state.irqsave_lock();
                    if let Some(allocation) = guard.as_mut()?.try_alloc_small(class).ok()? {
                        return Some(allocation.ptr);
                    }
                }
                let request = PageRequest {
                    pages: 1,
                    align: PAGE_SIZE,
                };
                let Some(candidate) = PendingPages::acquire(&self.source, request) else {
                    // Another CPU may have supplied a slab while we tried the source.
                    let mut guard = self.state.irqsave_lock();
                    return guard.as_mut()?.try_alloc_small(class).ok()?.map(|a| a.ptr);
                };
                // Keep the owner outside the lock scope on every commit branch.
                let mut pending = Some(candidate);
                let span = pending.as_ref().unwrap().span();
                // SAFETY: pending exclusively owns this unpublished page.
                unsafe { RawHeap::format_slab(span, class) }.ok()?;
                let outcome = {
                    let mut guard = self.state.irqsave_lock();
                    match guard.as_mut() {
                        Some(raw) => match unsafe { raw.commit_slab(span, class) } {
                            Ok(SlabCommit::Inserted(allocation)) => {
                                let _committed = pending.take().unwrap().commit();
                                Ok(allocation.ptr)
                            }
                            Ok(SlabCommit::Existing(allocation)) => Ok(allocation.ptr),
                            Err(error) => Err(error),
                        },
                        None => Err(HeapError::InvalidGeometry),
                    }
                };
                drop(pending);
                outcome.ok()
            }
            AllocationPlan::Pages(request) => {
                let mut pending = Some(PendingPages::acquire(&self.source, request)?);
                let span = pending.as_ref().unwrap().span();
                let outcome = {
                    let mut guard = self.state.irqsave_lock();
                    match guard.as_mut() {
                        Some(raw) => match unsafe { raw.commit_large(span, layout) } {
                            Ok(allocation) => {
                                let _committed = pending.take().unwrap().commit();
                                Ok(allocation.ptr)
                            }
                            Err(error) => Err(error),
                        },
                        None => Err(HeapError::InvalidGeometry),
                    }
                };
                drop(pending);
                outcome.ok()
            }
        }
    }

    /// # Safety
    /// `ptr` must be a live allocation from this heap with its original layout;
    /// the caller must relinquish all accesses to the allocation.
    pub unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.free(ptr, Some(layout)) };
    }

    /// # Safety
    /// `ptr` must be a live allocation from this heap and the caller must
    /// relinquish all accesses to it.
    pub unsafe fn deallocate_unknown_align(&self, ptr: *mut u8) {
        unsafe { self.free(ptr, None) };
    }

    unsafe fn free(&self, pointer: *mut u8, original_layout: Option<Layout>) {
        let pointer =
            NonNull::new(pointer).unwrap_or_else(|| fail_invalid_free(HeapError::NotAllocated));
        let action = {
            let mut guard = self.state.irqsave_lock();
            match guard.as_mut() {
                Some(raw) => unsafe { raw.free(pointer, original_layout) },
                None => Err(HeapError::InvalidGeometry),
            }
        };
        // Both source calls and the terminating diagnostic run after unlock.
        match action {
            Ok(ReleaseAction::None) => {}
            Ok(ReleaseAction::Pages(span)) => unsafe { self.source.release(span) },
            Err(error) => fail_invalid_free(error),
        }
    }

    pub fn heap_stats(&self) -> HeapStats {
        self.state
            .irqsave_lock()
            .as_ref()
            .map_or_else(HeapStats::default, RawHeap::stats)
    }
}

#[cold]
fn fail_invalid_free(error: HeapError) -> ! {
    // Kernel and rsrt panic handlers terminate; this never unwinds GlobalAlloc.
    panic!("invalid page heap free: {:?}", error)
}
