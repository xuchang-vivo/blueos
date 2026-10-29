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

#[path = "page_heap_support.rs"]
mod support;
use allocator_crate::page_heap::*;
use std::alloc::Layout;
use support::Fixture;

#[test]
fn large_run_keeps_all_actual_pages_and_rejects_interior_free() {
    assert_eq!(
        std::mem::size_of::<PageOwner>(),
        2 * std::mem::size_of::<usize>()
    );
    let mut f = Fixture::new(16);
    let span = f.take(4, 4 * PAGE_SIZE).unwrap();
    let layout = Layout::from_size_align(2 * PAGE_SIZE + 1, 4 * PAGE_SIZE).unwrap();
    let allocation = unsafe { f.heap.commit_large(span, layout) }.unwrap();
    assert_eq!(allocation.ptr, span.start());
    assert_eq!(allocation.capacity, 4 * PAGE_SIZE);
    let info = f.heap.allocation_info(allocation.ptr).unwrap();
    assert_eq!(info.kind, AllocationKind::Large);
    assert_eq!(info.requested_size, Some(layout.size()));
    assert_eq!(info.align, layout.align());
    assert_eq!(info.span, span);
    let interior = f
        .window
        .span(f.window.index_of(span.start()).unwrap() + 1, 1)
        .unwrap();
    assert_eq!(
        unsafe { f.heap.free(interior.start(), None) },
        Err(HeapError::InteriorPointer)
    );
    let within_head = unsafe { std::ptr::NonNull::new_unchecked(allocation.ptr.as_ptr().add(1)) };
    assert_eq!(
        unsafe { f.heap.free(within_head, None) },
        Err(HeapError::InteriorPointer)
    );
    assert_eq!(
        f.heap.stats(),
        HeapStats {
            large_pages: 4,
            backing_bytes: 16384,
            live_objects: 1,
            charged_bytes: 16384,
            peak_charged_bytes: 16384,
            ..HeapStats::default()
        }
    );
    match unsafe { f.heap.free(allocation.ptr, Some(layout)) }.unwrap() {
        ReleaseAction::Pages(detached) => f.release(detached),
        ReleaseAction::None => panic!("large allocation must release its run"),
    }
    assert_eq!(f.free_pages(), 16);
    assert_eq!(f.heap.stats().backing_bytes, 0);
    assert_eq!(f.heap.stats().peak_charged_bytes, 16384);
    assert_eq!(
        unsafe { f.heap.free(allocation.ptr, None) },
        Err(HeapError::NotAllocated)
    );
}

#[test]
fn classification_respects_alignment_and_rounds_page_requests() {
    for (size, align, capacity) in [
        (1, 1, 8),
        (8, 1, 8),
        (9, 8, 16),
        (17, 32, 32),
        (31, 64, 64),
        (65, 128, 128),
        (129, 256, 256),
        (257, 512, 512),
        (513, 1024, 1024),
        (1024, 8, 1024),
    ] {
        let AllocationPlan::Slab(class) =
            classify(Layout::from_size_align(size, align).unwrap()).unwrap()
        else {
            panic!("small request should use a slab");
        };
        assert_eq!(class.size(), capacity);
    }
    for (size, align, pages, page_align) in [
        (1025, 1, 1, 4096),
        (1, 2048, 1, 4096),
        (4097, 64, 2, 4096),
        (8193, 16384, 3, 16384),
    ] {
        assert_eq!(
            classify(Layout::from_size_align(size, align).unwrap()),
            Ok(AllocationPlan::Pages(PageRequest {
                pages,
                align: page_align
            }))
        );
    }
    assert_eq!(
        classify(Layout::from_size_align(0, 8).unwrap()),
        Err(HeapError::InvalidRequest)
    );
}

#[test]
fn geometry_rejects_invalid_bounds_and_accepts_object_addresses() {
    let f = Fixture::new(4);
    let base = f.window.base();
    let unaligned = unsafe { std::ptr::NonNull::new_unchecked(base.as_ptr().add(1)) };
    assert!(matches!(
        unsafe { PageWindow::new(unaligned, 1) },
        Err(HeapError::InvalidGeometry)
    ));
    assert!(matches!(
        unsafe { PageWindow::new(base, 0) },
        Err(HeapError::InvalidGeometry)
    ));
    assert!(matches!(
        unsafe { PageWindow::new(base, usize::MAX) },
        Err(HeapError::InvalidGeometry)
    ));
    assert!(matches!(
        unsafe { PageSpan::new(base, 0) },
        Err(HeapError::InvalidGeometry)
    ));
    assert!(matches!(
        unsafe { PageSpan::new(base, usize::MAX) },
        Err(HeapError::InvalidGeometry)
    ));
    assert_eq!(f.window.index_of(unaligned), Ok(0));
    assert_eq!(f.window.span(3, 2), Err(HeapError::OutOfWindow));
    assert_eq!(f.window.span(usize::MAX, 1), Err(HeapError::OutOfWindow));
    assert!(matches!(
        f.window.span(0, 0),
        Err(HeapError::InvalidGeometry)
    ));
    let end = unsafe { std::ptr::NonNull::new_unchecked(base.as_ptr().add(4 * PAGE_SIZE)) };
    assert_eq!(f.window.index_of(end), Err(HeapError::OutOfWindow));
    assert_eq!(f.heap.largest_free_slot(), 0);
    assert_eq!(f.heap.stats(), HeapStats::default());
    let mut short_owners = [PageOwner::EMPTY; 3];
    assert!(matches!(
        unsafe { PageHeap::new(f.window, &mut short_owners) },
        Err(HeapError::InvalidGeometry)
    ));
}

#[test]
fn invalid_large_commits_and_layouts_preserve_existing_allocation() {
    let mut f = Fixture::new(8);
    let span = f.take(2, PAGE_SIZE).unwrap();
    let layout = Layout::from_size_align(4097, 8).unwrap();
    let a = unsafe { f.heap.commit_large(span, layout) }.unwrap();
    let stats = f.heap.stats();
    assert_eq!(
        unsafe { f.heap.commit_large(span, layout) },
        Err(HeapError::Conflict)
    );
    let outside = unsafe { PageSpan::new(f.window.span(7, 1).unwrap().start(), 2) }.unwrap();
    assert_eq!(
        unsafe { f.heap.commit_large(outside, layout) },
        Err(HeapError::OutOfWindow)
    );
    let candidate = f.take(1, PAGE_SIZE).unwrap();
    let too_big = Layout::from_size_align(4097, 1).unwrap();
    assert_eq!(
        unsafe { f.heap.commit_large(candidate, too_big) },
        Err(HeapError::InvalidRequest)
    );
    let too_aligned = Layout::from_size_align(2048, 65536).unwrap();
    assert_eq!(
        unsafe { f.heap.commit_large(candidate, too_aligned) },
        Err(HeapError::InvalidRequest)
    );
    let zero = Layout::from_size_align(0, 1).unwrap();
    assert_eq!(
        unsafe { f.heap.commit_large(candidate, zero) },
        Err(HeapError::InvalidRequest)
    );
    let unaligned = unsafe { std::ptr::NonNull::new_unchecked(candidate.start().as_ptr().add(1)) };
    let unaligned_span = unsafe { PageSpan::new(unaligned, 1) };
    if let Ok(unaligned_span) = unaligned_span {
        assert_eq!(
            unsafe {
                f.heap
                    .commit_large(unaligned_span, Layout::from_size_align(2048, 1).unwrap())
            },
            Err(HeapError::InvalidGeometry)
        );
    } else {
        assert_eq!(unaligned_span, Err(HeapError::InvalidGeometry));
    }
    assert_eq!(
        unsafe {
            f.heap
                .free(a.ptr, Some(Layout::from_size_align(4098, 8).unwrap()))
        },
        Err(HeapError::LayoutMismatch)
    );
    assert_eq!(
        unsafe {
            f.heap
                .free(a.ptr, Some(Layout::from_size_align(4097, 16).unwrap()))
        },
        Err(HeapError::LayoutMismatch)
    );
    assert_eq!(
        unsafe { f.heap.free(candidate.start(), None) },
        Err(HeapError::NotAllocated)
    );
    assert_eq!(
        f.heap.allocation_info(candidate.start()).unwrap_err(),
        HeapError::NotAllocated
    );
    let other = Fixture::new(1);
    assert_eq!(
        unsafe { f.heap.free(other.window.base(), None) },
        Err(HeapError::NotAllocated)
    );
    assert_eq!(f.heap.stats(), stats);
    let info = f.heap.allocation_info(a.ptr).unwrap();
    assert_eq!(info.requested_size, Some(4097));
    assert_eq!(info.capacity, 8192);
    assert_eq!(info.align, 8);
    assert_eq!(info.span, span);
    f.release(candidate);
    match unsafe { f.heap.free(a.ptr, None) }.unwrap() {
        ReleaseAction::Pages(detached) => f.release(detached),
        ReleaseAction::None => panic!("large allocation must release its run"),
    }
    assert_eq!(f.free_pages(), 8);
}
