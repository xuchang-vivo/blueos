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

//! Raw metadata core for a heap backed by independently owned physical page runs.
//!
//! The caller acquires pages outside its heap lock and publishes them here. This
//! module never obtains pages, performs payload initialization, or releases pages.

mod metadata;

use core::{alloc::Layout, ptr::NonNull};
pub use metadata::PageOwner;

pub const PAGE_SIZE: usize = 4096;
pub const CLASS_SIZES: [usize; 8] = [8, 16, 32, 64, 128, 256, 512, 1024];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClassId(u8);

impl ClassId {
    pub fn size(self) -> usize {
        CLASS_SIZES[self.0 as usize]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageRequest {
    pub pages: usize,
    pub align: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationPlan {
    Slab(ClassId),
    Pages(PageRequest),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeapError {
    InvalidGeometry,
    InvalidRequest,
    OutOfWindow,
    Conflict,
    NotAllocated,
    InteriorPointer,
    LayoutMismatch,
    CorruptMetadata,
}

pub fn classify(layout: Layout) -> Result<AllocationPlan, HeapError> {
    if layout.size() == 0 {
        return Err(HeapError::InvalidRequest);
    }
    let required = layout.size().max(layout.align());
    if let Some(index) = CLASS_SIZES.iter().position(|size| *size >= required) {
        return Ok(AllocationPlan::Slab(ClassId(index as u8)));
    }
    let pages = layout
        .size()
        .checked_add(PAGE_SIZE - 1)
        .ok_or(HeapError::InvalidRequest)?
        / PAGE_SIZE;
    Ok(AllocationPlan::Pages(PageRequest {
        pages,
        align: layout.align().max(PAGE_SIZE),
    }))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageSpan {
    start: NonNull<u8>,
    pages: usize,
}

impl PageSpan {
    /// Describes a nonempty, page-aligned run without taking ownership of it.
    ///
    /// # Safety
    /// The pointer must retain the provenance needed to access any owned pages
    /// in the run. This descriptor does not make the described memory writable.
    pub unsafe fn new(start: NonNull<u8>, pages: usize) -> Result<Self, HeapError> {
        validate_geometry(start, pages)?;
        Ok(Self { start, pages })
    }

    pub fn start(self) -> NonNull<u8> {
        self.start
    }
    pub fn pages(self) -> usize {
        self.pages
    }
    pub fn byte_len(self) -> usize {
        self.pages * PAGE_SIZE
    }
}

#[derive(Clone, Copy)]
pub struct PageWindow {
    base: NonNull<u8>,
    pages: usize,
}

impl PageWindow {
    /// Describes a page-address window, which may include unavailable gaps.
    ///
    /// # Safety
    /// The base pointer must retain provenance for the pages that will be used.
    /// Neither constructing nor sharing a window permits accessing reserved or
    /// unowned pages, or accessing metadata without an exclusive heap borrow.
    pub unsafe fn new(base: NonNull<u8>, pages: usize) -> Result<Self, HeapError> {
        validate_geometry(base, pages)?;
        Ok(Self { base, pages })
    }

    pub fn base(self) -> NonNull<u8> {
        self.base
    }
    pub fn pages(self) -> usize {
        self.pages
    }

    pub fn span(self, first: usize, pages: usize) -> Result<PageSpan, HeapError> {
        let end = first.checked_add(pages).ok_or(HeapError::OutOfWindow)?;
        if end > self.pages {
            return Err(HeapError::OutOfWindow);
        }
        // wrapping_add preserves provenance and does not assert that gaps in
        // this address window belong to a single allocated Rust object.
        let start = self.base.as_ptr().wrapping_add(first * PAGE_SIZE);
        unsafe {
            PageSpan::new(
                NonNull::new(start).ok_or(HeapError::InvalidGeometry)?,
                pages,
            )
        }
    }

    /// Returns the covering page for either a page or an object address.
    pub fn index_of(self, ptr: NonNull<u8>) -> Result<usize, HeapError> {
        let offset = ptr
            .as_ptr()
            .addr()
            .checked_sub(self.base.as_ptr().addr())
            .ok_or(HeapError::OutOfWindow)?;
        if offset >= self.pages * PAGE_SIZE {
            return Err(HeapError::OutOfWindow);
        }
        Ok(offset / PAGE_SIZE)
    }
}

// These types only describe addresses. Sharing them does not grant access to
// payloads or metadata; metadata mutations still require &mut PageHeap.
unsafe impl Send for PageWindow {}
unsafe impl Sync for PageWindow {}
unsafe impl Send for PageSpan {}
unsafe impl Sync for PageSpan {}

fn validate_geometry(base: NonNull<u8>, pages: usize) -> Result<(), HeapError> {
    let bytes = pages
        .checked_mul(PAGE_SIZE)
        .ok_or(HeapError::InvalidGeometry)?;
    if pages == 0
        || base.as_ptr().addr() & (PAGE_SIZE - 1) != 0
        || base.as_ptr().addr().checked_add(bytes).is_none()
    {
        return Err(HeapError::InvalidGeometry);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Allocation {
    pub ptr: NonNull<u8>,
    pub capacity: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationKind {
    Slab(ClassId),
    Large,
}

#[derive(Clone, Copy, Debug)]
pub struct AllocationInfo {
    pub kind: AllocationKind,
    pub requested_size: Option<usize>,
    pub capacity: usize,
    pub align: usize,
    pub span: PageSpan,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HeapStats {
    pub slab_pages: usize,
    pub large_pages: usize,
    pub backing_bytes: usize,
    pub live_objects: usize,
    pub charged_bytes: usize,
    pub peak_charged_bytes: usize,
}

#[must_use = "detached physical pages must be returned to their source"]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReleaseAction {
    None,
    Pages(PageSpan),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlabCommit {
    Inserted(Allocation),
    Existing(Allocation),
}

pub struct PageHeap<'a> {
    window: PageWindow,
    owners: &'a mut [PageOwner],
    stats: HeapStats,
}

impl<'a> PageHeap<'a> {
    /// Initializes a new ownership table for the address window.
    ///
    /// # Safety
    /// No live heap allocations may exist in the window. The caller must keep
    /// the window's actual owned pages valid until detached, and the ownership
    /// table must not overlap any page that may be published to this heap.
    pub unsafe fn new(window: PageWindow, owners: &'a mut [PageOwner]) -> Result<Self, HeapError> {
        if owners.len() < window.pages
            || !PageOwner::can_encode_pages(window.pages)
            || !PageOwner::can_encode_index(window.pages - 1)
        {
            return Err(HeapError::InvalidGeometry);
        }
        let owners = &mut owners[..window.pages];
        owners.fill(PageOwner::EMPTY);
        Ok(Self {
            window,
            owners,
            stats: HeapStats::default(),
        })
    }

    pub fn stats(&self) -> HeapStats {
        self.stats
    }

    /// Committed free slab slots; unowned window pages are not heap storage.
    pub fn largest_free_slot(&self) -> usize {
        0
    }

    /// Publishes an exclusively owned run without adding an in-band header.
    ///
    /// # Safety
    /// The caller exclusively owns all pages in the run and must retain their
    /// ownership after success until this heap returns the run from `free`.
    /// The run must be physically contiguous and writable for its entire size.
    pub unsafe fn commit_large(
        &mut self,
        span: PageSpan,
        layout: Layout,
    ) -> Result<Allocation, HeapError> {
        if span.start.as_ptr().addr() & (PAGE_SIZE - 1) != 0 {
            return Err(HeapError::InvalidGeometry);
        }
        let first = self.window.index_of(span.start)?;
        let end = first
            .checked_add(span.pages)
            .ok_or(HeapError::OutOfWindow)?;
        let target = self.owners.get(first..end).ok_or(HeapError::OutOfWindow)?;
        if !target.iter().all(PageOwner::is_empty) {
            return Err(HeapError::Conflict);
        }
        let head = PageOwner::large_head(span.pages, layout)?;
        if layout.size() > span.byte_len() || span.start.as_ptr().addr() & (layout.align() - 1) != 0
        {
            return Err(HeapError::InvalidRequest);
        }
        let tail = PageOwner::large_tail(first)?;
        let mut stats = self.stats;
        stats.large_pages = stats
            .large_pages
            .checked_add(span.pages)
            .ok_or(HeapError::InvalidRequest)?;
        stats.backing_bytes = stats
            .backing_bytes
            .checked_add(span.byte_len())
            .ok_or(HeapError::InvalidRequest)?;
        stats.live_objects = stats
            .live_objects
            .checked_add(1)
            .ok_or(HeapError::InvalidRequest)?;
        stats.charged_bytes = stats
            .charged_bytes
            .checked_add(span.byte_len())
            .ok_or(HeapError::InvalidRequest)?;
        stats.peak_charged_bytes = stats.peak_charged_bytes.max(stats.charged_bytes);
        self.owners[first] = head;
        self.owners[first + 1..end].fill(tail);
        self.stats = stats;
        Ok(Allocation {
            ptr: span.start,
            capacity: span.byte_len(),
        })
    }

    pub fn allocation_info(&self, ptr: NonNull<u8>) -> Result<AllocationInfo, HeapError> {
        let index = self
            .window
            .index_of(ptr)
            .map_err(|_| HeapError::NotAllocated)?;
        let owner = self.owners[index];
        if owner.is_empty() {
            return Err(HeapError::NotAllocated);
        }
        let first = match owner.tail_index()? {
            Some(first) if first < index => first,
            Some(_) => return Err(HeapError::CorruptMetadata),
            None => index,
        };
        let (pages, layout) = self
            .owners
            .get(first)
            .ok_or(HeapError::CorruptMetadata)?
            .head()?;
        let end = first.checked_add(pages).ok_or(HeapError::CorruptMetadata)?;
        let chain = self
            .owners
            .get(first + 1..end)
            .ok_or(HeapError::CorruptMetadata)?;
        let tail = PageOwner::large_tail(first).map_err(|_| HeapError::CorruptMetadata)?;
        if !chain.iter().all(|owner| owner.same_encoding(tail)) || index >= end {
            return Err(HeapError::CorruptMetadata);
        }
        let span = self
            .window
            .span(first, pages)
            .map_err(|_| HeapError::CorruptMetadata)?;
        if layout.size() > span.byte_len() || span.start.as_ptr().addr() & (layout.align() - 1) != 0
        {
            return Err(HeapError::CorruptMetadata);
        }
        if ptr != span.start {
            return Err(HeapError::InteriorPointer);
        }
        Ok(AllocationInfo {
            kind: AllocationKind::Large,
            requested_size: Some(layout.size()),
            capacity: span.byte_len(),
            align: layout.align(),
            span,
        })
    }

    /// Validates and detaches a large run. The caller returns detached pages to
    /// their source after releasing any heap lock.
    ///
    /// # Safety
    /// On success no payload access may remain, and the caller must honor the
    /// returned release action. `ptr` must retain its original provenance.
    pub unsafe fn free(
        &mut self,
        ptr: NonNull<u8>,
        layout: Option<Layout>,
    ) -> Result<ReleaseAction, HeapError> {
        let info = self.allocation_info(ptr)?;
        if let Some(layout) = layout {
            if Some(layout.size()) != info.requested_size || layout.align() != info.align {
                return Err(HeapError::LayoutMismatch);
            }
        }
        let first = self.window.index_of(info.span.start)?;
        let mut stats = self.stats;
        stats.large_pages = stats
            .large_pages
            .checked_sub(info.span.pages)
            .ok_or(HeapError::CorruptMetadata)?;
        stats.backing_bytes = stats
            .backing_bytes
            .checked_sub(info.capacity)
            .ok_or(HeapError::CorruptMetadata)?;
        stats.live_objects = stats
            .live_objects
            .checked_sub(1)
            .ok_or(HeapError::CorruptMetadata)?;
        stats.charged_bytes = stats
            .charged_bytes
            .checked_sub(info.capacity)
            .ok_or(HeapError::CorruptMetadata)?;
        self.owners[first..first + info.span.pages].fill(PageOwner::EMPTY);
        self.stats = stats;
        Ok(ReleaseAction::Pages(info.span))
    }
}
