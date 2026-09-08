//! Host-to-device transfer primitives shared by paged weights and elastic arena clients.
//!
//! Callers name an opaque device target. Whether the target is CPU-mapped is handled here:
//! mapped targets take the direct write fast path, while ordinary device-local targets use a
//! host-visible staging allocation and `vkCmdCopyBuffer`.

use std::sync::{Arc, RwLock};

use ash::vk;
use gpu_allocator::MemoryLocation;

use infr_core::backend::Buffer;
use infr_core::error::Result;
use infr_core::pager_profile;

use crate::{as_vk_buf, be, copy_to_mapped, ImportedHostAllocation, VkBuffer, VulkanBackend};

/// One source/destination group submitted to the optional dedicated transfer queue. Both owners
/// are retained by the queue's command slot until its timeline value completes.
pub(crate) struct TransferCopyBatch {
    pub(crate) src: Arc<dyn Buffer>,
    pub(crate) dst: Arc<dyn Buffer>,
    pub(crate) regions: Vec<vk::BufferCopy>,
}

const DEDICATED_TRANSFER_COMMAND_SLOTS: usize = 3;

struct DedicatedTransferCommandSlot {
    cmd: vk::CommandBuffer,
    pending_value: u64,
    keepalive: Vec<Arc<dyn Buffer>>,
}

/// Transfer-family stream for imported host RAM to device-arena copies. The timeline value is
/// consumed by the main queue submission that first reads the uploaded experts.
pub(crate) struct DedicatedTransferQueue {
    queue: vk::Queue,
    family_index: u32,
    pool: vk::CommandPool,
    timeline: vk::Semaphore,
    slots: Vec<DedicatedTransferCommandSlot>,
    cursor: usize,
    next_value: u64,
}

impl DedicatedTransferQueue {
    pub(crate) fn new(device: &ash::Device, family_index: u32) -> Result<Self> {
        let queue = unsafe { device.get_device_queue(family_index, 0) };
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|error| be(format!("create dedicated transfer command pool: {error}")))?;
        let commands = match unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(DEDICATED_TRANSFER_COMMAND_SLOTS as u32),
            )
        } {
            Ok(commands) => commands,
            Err(error) => {
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(be(format!(
                    "allocate dedicated transfer command buffers: {error}"
                )));
            }
        };
        let mut semaphore_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline = match unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type),
                None,
            )
        } {
            Ok(semaphore) => semaphore,
            Err(error) => {
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(be(format!(
                    "create dedicated transfer timeline semaphore: {error}"
                )));
            }
        };
        Ok(Self {
            queue,
            family_index,
            pool,
            timeline,
            slots: commands
                .into_iter()
                .map(|cmd| DedicatedTransferCommandSlot {
                    cmd,
                    pending_value: 0,
                    keepalive: Vec::new(),
                })
                .collect(),
            cursor: 0,
            next_value: 1,
        })
    }

    pub(crate) fn family_index(&self) -> u32 {
        self.family_index
    }

    pub(crate) fn timeline(&self) -> vk::Semaphore {
        self.timeline
    }

    pub(crate) fn submit(
        &mut self,
        shared: &crate::VulkanShared,
        batches: &[TransferCopyBatch],
    ) -> Result<u64> {
        debug_assert!(!batches.is_empty());
        let slot = &mut self.slots[self.cursor];
        if slot.pending_value != 0 {
            let semaphores = [self.timeline];
            let values = [slot.pending_value];
            unsafe {
                shared.device.wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&semaphores)
                        .values(&values),
                    u64::MAX,
                )
            }
            .map_err(|error| be(format!("wait reusable transfer command slot: {error}")))?;
            slot.keepalive.clear();
        }
        unsafe {
            shared
                .device
                .reset_command_buffer(slot.cmd, vk::CommandBufferResetFlags::empty())
        }
        .map_err(|error| be(format!("reset dedicated transfer command buffer: {error}")))?;
        unsafe {
            shared.device.begin_command_buffer(
                slot.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|error| be(format!("begin dedicated transfer command buffer: {error}")))?;

        let host_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        unsafe {
            shared.device.cmd_pipeline_barrier(
                slot.cmd,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[host_barrier],
                &[],
                &[],
            );
            for batch in batches {
                let src = as_vk_buf(batch.src.as_ref())?.buffer;
                let dst = as_vk_buf(batch.dst.as_ref())?.buffer;
                shared
                    .device
                    .cmd_copy_buffer(slot.cmd, src, dst, &batch.regions);
            }
            shared
                .device
                .end_command_buffer(slot.cmd)
                .map_err(|error| be(format!("end dedicated transfer command buffer: {error}")))?;
        }

        let value = self.next_value;
        self.next_value = self
            .next_value
            .checked_add(1)
            .ok_or_else(|| be("dedicated transfer timeline value overflow"))?;
        let commands = [slot.cmd];
        let signals = [self.timeline];
        let signal_values = [value];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .signal_semaphores(&signals)
            .push_next(&mut timeline);
        shared.submit_dedicated_transfer(self.queue, &submit)?;

        slot.pending_value = value;
        slot.keepalive.reserve(batches.len() * 2);
        for batch in batches {
            slot.keepalive.push(Arc::clone(&batch.src));
            slot.keepalive.push(Arc::clone(&batch.dst));
        }
        self.cursor = (self.cursor + 1) % self.slots.len();
        Ok(value)
    }

    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_semaphore(self.timeline, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Backend contract consumed by residency logic. It exposes data movement, not Vulkan memory
/// types or queue choices; a different executor can satisfy the same requests without changing
/// pager policy.
pub(crate) trait TransferExecutor: Sync {
    fn materialize_staging(&self, src: &[u8]) -> Result<(Arc<dyn Buffer>, usize)>;

    fn complete_copies_now(
        &self,
        copies: &[(Arc<dyn Buffer>, usize, DeviceTransferTarget, usize)],
    ) -> Result<()>;

    /// Relocate immutable device ranges as one ordered transaction. Residency policy decides
    /// which ranges move; the executor owns queue choice, barriers and completion semantics.
    fn relocate_device_ranges_now(
        &self,
        copies: &[(DeviceTransferTarget, DeviceTransferTarget)],
    ) -> Result<()>;

    fn fill_target_now(
        &self,
        target: &DeviceTransferTarget,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<()>;
}

/// Transport bindings for one loaded model session. Hardware capabilities are probed while the
/// Vulkan backend is created and host allocations are imported at session finalization. Runtime
/// uploads only locate their source inside these ranges and execute the established target route.
///
/// The table is normally immutable. Its write side exists solely for the rare queue-submit OOM
/// recovery path, which retires unused imported tails after a queue drain. Prepared/in-flight
/// copies retain their source buffers independently, so shrinking the table cannot invalidate a
/// command that has already been recorded.
#[derive(Default)]
pub(crate) struct SessionTransferPlan {
    imports: RwLock<Vec<ImportedHostAllocation>>,
}

impl SessionTransferPlan {
    pub(crate) fn new(imports: Vec<ImportedHostAllocation>) -> Self {
        Self {
            imports: RwLock::new(imports),
        }
    }

    /// Add one host-to-device request to `prepared`. Imported host ranges become Vulkan buffer
    /// copies, mapped targets are filled immediately, and every other target is staged. Those are
    /// backend details: callers provide only an opaque source slice and device target.
    pub(crate) fn prepare_upload<E: TransferExecutor>(
        &self,
        executor: &E,
        src: &[u8],
        target: &DeviceTransferTarget,
        prepared: &mut PreparedTransfer,
    ) -> Result<()> {
        if src.len() != target.len() {
            return Err(be(format!(
                "host upload has {} bytes but its device target has {}",
                src.len(),
                target.len()
            )));
        }
        if self.append_imported(src, target, prepared) {
            return Ok(());
        }
        if let Some(dst) = target.mapped_ptr() {
            let started = pager_profile::active().then(std::time::Instant::now);
            parallel_copy_to_mapped(src, dst);
            if let Some(t0) = started {
                pager_profile::record_memcpy(src.len(), t0.elapsed());
            }
            return Ok(());
        }
        let (source, source_ptr) = executor.materialize_staging(src)?;
        prepared.copies.push(PreparedCopy {
            source,
            source_offset: 0,
            source_ptr,
            target: target.clone(),
            len: src.len(),
            dedicated: false,
        });
        Ok(())
    }

    fn append_imported(
        &self,
        src: &[u8],
        target: &DeviceTransferTarget,
        prepared: &mut PreparedTransfer,
    ) -> bool {
        let Some(ranges) = self.imported_ranges(src) else {
            return false;
        };
        let mut advanced = 0usize;
        for range in ranges {
            prepared.copies.push(PreparedCopy {
                source: range.buffer,
                source_offset: range.offset,
                source_ptr: unsafe { src.as_ptr().add(advanced) } as usize,
                target: target
                    .subtarget(advanced, range.len)
                    .expect("imported source range was validated against its device target"),
                len: range.len,
                dedicated: true,
            });
            advanced += range.len;
        }
        debug_assert_eq!(advanced, src.len());
        true
    }

    fn imported_ranges(&self, src: &[u8]) -> Option<Vec<crate::ImportedHostRange>> {
        self.imports
            .read()
            .unwrap()
            .iter()
            .find(|import| import.contains(src.as_ptr(), src.len()))
            .and_then(|import| import.ranges(src.as_ptr(), src.len()))
    }

    /// Release up to `target_bytes` of imported-host aliases without touching any source buffer
    /// retained by a prepared or in-flight command. Imports cover contiguous prefixes, so only a
    /// physical tail shard can be retired while keeping all earlier source addresses valid.
    pub(crate) fn shed_unused_import_tails(&self, target_bytes: usize) -> usize {
        let mut imports = self.imports.write().unwrap();
        let mut released = 0usize;
        while released < target_bytes {
            let candidate = imports
                .iter()
                .enumerate()
                .filter_map(|(index, import)| {
                    let shard = import.shards.last()?;
                    (Arc::strong_count(&shard.buffer) == 1).then_some((index, shard.len))
                })
                .max_by_key(|&(_, len)| len);
            let Some((index, _)) = candidate else {
                break;
            };
            let shard = imports[index]
                .shards
                .pop()
                .expect("candidate import has a tail shard");
            debug_assert_eq!(
                shard.offset.saturating_add(shard.len),
                imports[index].imported_len
            );
            imports[index].imported_len = shard.offset;
            released = released.saturating_add(shard.len);
            drop(shard);
        }
        imports.retain(|import| !import.shards.is_empty());
        released
    }

    pub(crate) fn upload_now<E: TransferExecutor>(
        &self,
        executor: &E,
        src: &[u8],
        target: &DeviceTransferTarget,
    ) -> Result<()> {
        let mut prepared = PreparedTransfer::default();
        self.prepare_upload(executor, src, target, &mut prepared)?;
        prepared.complete_now(executor)
    }

    /// Whether a target can be filled by the dedicated host worker without touching a Vulkan
    /// queue. The scheduler sees only this execution property, never the physical backing type.
    pub(crate) fn supports_host_worker(&self, target: &DeviceTransferTarget) -> bool {
        target.mapped_ptr().is_some()
    }

    pub(crate) fn copy_on_host_worker(
        &self,
        src: &[u8],
        target: &DeviceTransferTarget,
    ) -> Result<()> {
        if src.len() != target.len() {
            return Err(be("host-worker copy size does not match its target"));
        }
        let dst = target
            .mapped_ptr()
            .ok_or_else(|| be("host worker cannot fill this device target"))?;
        let started = pager_profile::active().then(std::time::Instant::now);
        parallel_copy_to_mapped(src, dst);
        if let Some(t0) = started {
            pager_profile::record_memcpy(src.len(), t0.elapsed());
        }
        Ok(())
    }

    pub(crate) fn fill_on_host_worker(
        &self,
        target: &DeviceTransferTarget,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        let ptr = target
            .mapped_ptr()
            .ok_or_else(|| be("host worker cannot fill this device target"))?;
        let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, target.len()) };
        fill(bytes)
    }

    pub(crate) fn fill_now<E: TransferExecutor>(
        &self,
        executor: &E,
        target: &DeviceTransferTarget,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        executor.fill_target_now(target, fill)
    }
}

struct PreparedCopy {
    source: Arc<dyn Buffer>,
    source_offset: usize,
    source_ptr: usize,
    target: DeviceTransferTarget,
    len: usize,
    /// Imported host memory and unified device arenas can use the session's transfer-family
    /// queue. Temporary staging remains exclusive to the main queue.
    dedicated: bool,
}

/// One backend-resolved transfer batch. The source route and staging ownership are frozen before
/// this value reaches the scheduler; it only records the batch into a command stream or completes
/// it immediately when no recorder is available.
#[derive(Default)]
pub(crate) struct PreparedTransfer {
    copies: Vec<PreparedCopy>,
}

impl PreparedTransfer {
    pub(crate) fn append(&mut self, mut other: Self) {
        self.copies.append(&mut other.copies);
    }

    pub(crate) fn record(mut self, rec: &crate::Recorder<'_>) -> Result<()> {
        let mut dedicated_groups: Vec<TransferCopyBatch> = Vec::new();
        let mut main_groups: Vec<TransferCopyBatch> = Vec::new();
        for copy in &self.copies {
            let groups = if copy.dedicated {
                &mut dedicated_groups
            } else {
                &mut main_groups
            };
            let src_handle = as_vk_buf(copy.source.as_ref())?.buffer;
            let dst_handle = as_vk_buf(copy.target.buffer())?.buffer;
            let group = match groups.iter_mut().find(|group| {
                as_vk_buf(group.src.as_ref()).is_ok_and(|buf| buf.buffer == src_handle)
                    && as_vk_buf(group.dst.as_ref()).is_ok_and(|buf| buf.buffer == dst_handle)
            }) {
                Some(group) => group,
                None => {
                    groups.push(TransferCopyBatch {
                        src: Arc::clone(&copy.source),
                        dst: copy.target.buffer_arc(),
                        regions: Vec::new(),
                    });
                    groups.last_mut().expect("group was just appended")
                }
            };
            group.regions.push(
                vk::BufferCopy::default()
                    .src_offset(copy.source_offset as u64)
                    .dst_offset(copy.target.buffer_offset() as u64)
                    .size(copy.len as u64),
            );
        }
        if !dedicated_groups.is_empty() && !rec.submit_dedicated_transfer(&dedicated_groups)? {
            main_groups.append(&mut dedicated_groups);
        }
        if !main_groups.is_empty() {
            rec.host_transfer_barrier();
            for group in &main_groups {
                rec.retain_buffer(Arc::clone(&group.src));
                rec.retain_buffer(Arc::clone(&group.dst));
                rec.copy_regions(group.src.as_ref(), group.dst.as_ref(), &group.regions);
            }
        }
        if pager_profile::active() {
            for copy in &self.copies {
                pager_profile::record_gpu_copy(copy.len);
            }
        }
        self.copies.clear();
        Ok(())
    }

    pub(crate) fn complete_now<E: TransferExecutor>(mut self, executor: &E) -> Result<()> {
        if self.copies.is_empty() {
            return Ok(());
        }
        let started = pager_profile::active().then(std::time::Instant::now);
        let mut bytes = 0usize;
        let mut staged = Vec::new();
        for copy in self.copies.drain(..) {
            if let Some(dst) = copy.target.mapped_ptr() {
                let src =
                    unsafe { std::slice::from_raw_parts(copy.source_ptr as *const u8, copy.len) };
                parallel_copy_to_mapped(src, dst);
                bytes = bytes.saturating_add(copy.len);
            } else {
                staged.push((copy.source, copy.source_offset, copy.target, copy.len));
            }
        }
        if let Some(t0) = started {
            pager_profile::record_memcpy(bytes, t0.elapsed());
        }
        executor.complete_copies_now(&staged)
    }
}

/// Parallel host copy used by direct host-visible transfer endpoints. Kept in the transport
/// backend so residency policy never handles raw mapped pointers.
pub(crate) fn parallel_copy_to_mapped(src: &[u8], dst: *mut u8) {
    use rayon::prelude::*;
    const CHUNK: usize = 4 << 20;
    if src.len() <= CHUNK {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    let dst_addr = dst as usize;
    src.par_chunks(CHUNK)
        .enumerate()
        .for_each(|(i, chunk)| unsafe {
            std::ptr::copy_nonoverlapping(
                chunk.as_ptr(),
                (dst_addr + i * CHUNK) as *mut u8,
                chunk.len(),
            );
        });
}

/// A byte range that can be consumed by Vulkan. `mapped_ptr` points at the start of this exact
/// range when the physical backing is host-visible; it is absent for ordinary device-local VRAM.
#[derive(Clone)]
pub(crate) struct DeviceTransferTarget {
    buffer: Arc<dyn Buffer>,
    /// Offset relative to this logical `Buffer` handle (used by `Recorder`, which adds sub_offset).
    buffer_offset: usize,
    /// Absolute offset in the underlying VkBuffer (used by raw one-shot submissions).
    vk_offset: usize,
    mapped_ptr: Option<usize>,
    len: usize,
}

impl DeviceTransferTarget {
    pub(crate) fn new(buffer: Arc<dyn Buffer>, offset: usize, len: usize) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| be("device transfer target range overflow"))?;
        if end > buffer.len_bytes() {
            return Err(be(format!(
                "device transfer target {offset}..{end} exceeds its {}-byte buffer",
                buffer.len_bytes()
            )));
        }
        let vk_buffer = as_vk_buf(buffer.as_ref())?;
        let vk_offset = vk_buffer
            .sub_offset
            .checked_add(offset)
            .ok_or_else(|| be("device transfer target Vulkan offset overflow"))?;
        let mapped_ptr = vk_buffer
            .mapped_ptr()
            .map(|ptr| unsafe { ptr.add(offset) } as usize);
        Ok(Self {
            buffer,
            buffer_offset: offset,
            vk_offset,
            mapped_ptr,
            len,
        })
    }

    pub(crate) fn buffer(&self) -> &dyn Buffer {
        self.buffer.as_ref()
    }

    pub(crate) fn buffer_arc(&self) -> Arc<dyn Buffer> {
        Arc::clone(&self.buffer)
    }

    pub(crate) fn buffer_offset(&self) -> usize {
        self.buffer_offset
    }

    pub(crate) fn vk_offset(&self) -> usize {
        self.vk_offset
    }

    pub(crate) fn mapped_ptr(&self) -> Option<*mut u8> {
        self.mapped_ptr.map(|ptr| ptr as *mut u8)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn subtarget(&self, offset: usize, len: usize) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| be("device transfer sub-range overflow"))?;
        if end > self.len {
            return Err(be("device transfer sub-range exceeds its parent target"));
        }
        Ok(Self {
            buffer: Arc::clone(&self.buffer),
            buffer_offset: self
                .buffer_offset
                .checked_add(offset)
                .ok_or_else(|| be("device transfer sub-range Vulkan offset overflow"))?,
            vk_offset: self
                .vk_offset
                .checked_add(offset)
                .ok_or_else(|| be("device transfer sub-range Vulkan offset overflow"))?,
            mapped_ptr: self.mapped_ptr.map(|ptr| ptr + offset),
            len,
        })
    }
}

impl VulkanBackend {
    /// Allocate staging explicitly on the host-visible non-device-local heap when the device
    /// exposes one. This prevents a small RDNA2 ReBAR heap from being consumed by the fallback
    /// that exists precisely because the expert arena did not fit that heap.
    fn make_host_transfer_buffer(&self, size: usize) -> Result<VkBuffer> {
        let Some(memory_type) = self.shared.host_overflow_type else {
            return self.make_buf(size, MemoryLocation::CpuToGpu, "host-transfer-staging");
        };
        let info = vk::BufferCreateInfo::default()
            .size(crate::fill_span(size))
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.shared.device.create_buffer(&info, None) }
            .map_err(|error| be(format!("create_buffer(host-transfer-staging): {error}")))?;
        let requirements = unsafe { self.shared.device.get_buffer_memory_requirements(buffer) };
        if requirements.memory_type_bits & (1 << memory_type) == 0 {
            unsafe { self.shared.device.destroy_buffer(buffer, None) };
            return self.make_buf(size, MemoryLocation::CpuToGpu, "host-transfer-staging");
        }
        self.alloc_vram_mapped(buffer, size, &requirements, memory_type, true, false, false)
            .inspect_err(|_| unsafe { self.shared.device.destroy_buffer(buffer, None) })
    }

    /// Materialize bytes in a temporary host-visible Vulkan buffer without submitting work. The
    /// returned owner must stay alive until every command that reads it has completed.
    fn stage_host_bytes(&self, src: &[u8]) -> Result<(Arc<dyn Buffer>, usize)> {
        let staging = self.make_host_transfer_buffer(src.len())?;
        let ptr = staging
            .mapped_ptr()
            .ok_or_else(|| be("host transfer staging allocation is not mapped"))?;
        let started = pager_profile::active().then(std::time::Instant::now);
        copy_to_mapped(src, ptr);
        if let Some(t0) = started {
            pager_profile::record_memcpy(src.len(), t0.elapsed());
        }
        Ok((Arc::new(staging), ptr as usize))
    }

    /// Fill one target range. The callback always receives writable host memory, either the final
    /// mapped destination or a temporary staging allocation. The staged path is synchronous; it
    /// is the universal correctness fallback used only when a direct/imported batch is unavailable.
    fn write_device_target(
        &self,
        target: &DeviceTransferTarget,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        if let Some(ptr) = target.mapped_ptr() {
            let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, target.len()) };
            fill(bytes)?;
            return Ok(());
        }

        let staging = self.make_host_transfer_buffer(target.len())?;
        let staging_ptr = staging
            .mapped_ptr()
            .ok_or_else(|| be("host transfer staging allocation is not mapped"))?;
        let staging_bytes = unsafe { std::slice::from_raw_parts_mut(staging_ptr, target.len()) };
        fill(staging_bytes)?;
        let source: Arc<dyn Buffer> = Arc::new(staging);
        self.copy_transfer_targets_now(&[(source, 0, target.clone(), target.len())])?;
        Ok(())
    }

    /// Execute already-materialized buffer copies immediately on the main queue. Used by the
    /// Decode overlap compatibility path when no ambient recorder exists; mapped destinations can
    /// still avoid this through their direct CPU fallback.
    fn copy_transfer_targets_now(
        &self,
        copies: &[(Arc<dyn Buffer>, usize, DeviceTransferTarget, usize)],
    ) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let mut resolved = Vec::with_capacity(copies.len());
        for (source, source_offset, target, len) in copies {
            if *len > target.len() {
                return Err(be("immediate transfer exceeds its device target"));
            }
            resolved.push((
                as_vk_buf(source.as_ref())?.buffer,
                *source_offset as u64,
                as_vk_buf(target.buffer())?.buffer,
                target.vk_offset() as u64,
                *len as u64,
            ));
        }
        let shared = Arc::clone(&self.shared);
        self.one_shot(move |cmd| unsafe {
            let host = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            shared.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[host],
                &[],
                &[],
            );
            for &(src, src_offset, dst, dst_offset, len) in &resolved {
                shared.device.cmd_copy_buffer(
                    cmd,
                    src,
                    dst,
                    &[vk::BufferCopy::default()
                        .src_offset(src_offset)
                        .dst_offset(dst_offset)
                        .size(len)],
                );
            }
        })?;
        if pager_profile::active() {
            let bytes = copies
                .iter()
                .fold(0usize, |total, copy| total.saturating_add(copy.3));
            pager_profile::record_gpu_copy(bytes);
        }
        Ok(())
    }

    /// Move device-resident ranges without exposing Vulkan queue or memory details to the pager.
    /// This is a rare ownership-boundary operation (KV growth / phase arena claim), so one
    /// synchronous batched submission is preferable to per-slot submissions or CPU round-trips.
    fn relocate_device_targets_now(
        &self,
        copies: &[(DeviceTransferTarget, DeviceTransferTarget)],
    ) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }

        struct Group {
            src: vk::Buffer,
            dst: vk::Buffer,
            regions: Vec<vk::BufferCopy>,
        }

        let mut groups: Vec<Group> = Vec::new();
        let mut bytes = 0usize;
        for (source, target) in copies {
            if source.len() != target.len() {
                return Err(be("device relocation source and target sizes differ"));
            }
            let src = as_vk_buf(source.buffer())?.buffer;
            let dst = as_vk_buf(target.buffer())?.buffer;
            let src_start = source.vk_offset();
            let src_end = src_start
                .checked_add(source.len())
                .ok_or_else(|| be("device relocation source range overflow"))?;
            let dst_start = target.vk_offset();
            let dst_end = dst_start
                .checked_add(target.len())
                .ok_or_else(|| be("device relocation target range overflow"))?;
            if src == dst && src_start < dst_end && dst_start < src_end {
                return Err(be("device relocation ranges overlap in one Vulkan buffer"));
            }
            let group = match groups
                .iter_mut()
                .find(|group| group.src == src && group.dst == dst)
            {
                Some(group) => group,
                None => {
                    groups.push(Group {
                        src,
                        dst,
                        regions: Vec::new(),
                    });
                    groups.last_mut().expect("group was just appended")
                }
            };
            group.regions.push(
                vk::BufferCopy::default()
                    .src_offset(src_start as u64)
                    .dst_offset(dst_start as u64)
                    .size(source.len() as u64),
            );
            bytes = bytes.saturating_add(source.len());
        }

        let shared = Arc::clone(&self.shared);
        self.one_shot(move |cmd| unsafe {
            // The execution gate prevents new arena users while this transaction runs. These
            // barriers close the write-after-read hazard against already-submitted expert reads
            // and publish the relocated bytes to subsequent compute submissions.
            let before = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
            shared.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            for group in &groups {
                shared
                    .device
                    .cmd_copy_buffer(cmd, group.src, group.dst, &group.regions);
            }
            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ);
            shared.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        })?;
        if tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!(
                relocated_ranges = copies.len(),
                relocated_bytes = bytes,
                "compacted device-resident ranges for a unified VRAM claim"
            );
        }
        Ok(())
    }
}

impl TransferExecutor for VulkanBackend {
    fn materialize_staging(&self, src: &[u8]) -> Result<(Arc<dyn Buffer>, usize)> {
        VulkanBackend::stage_host_bytes(self, src)
    }

    fn complete_copies_now(
        &self,
        copies: &[(Arc<dyn Buffer>, usize, DeviceTransferTarget, usize)],
    ) -> Result<()> {
        VulkanBackend::copy_transfer_targets_now(self, copies)
    }

    fn relocate_device_ranges_now(
        &self,
        copies: &[(DeviceTransferTarget, DeviceTransferTarget)],
    ) -> Result<()> {
        VulkanBackend::relocate_device_targets_now(self, copies)
    }

    fn fill_target_now(
        &self,
        target: &DeviceTransferTarget,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        VulkanBackend::write_device_target(self, target, fill)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use infr_core::backend::Buffer;

    use super::SessionTransferPlan;
    use crate::{ImportedHostAllocation, ImportedHostShard};

    struct DummyBuffer(usize);

    impl Buffer for DummyBuffer {
        fn len_bytes(&self) -> usize {
            self.0
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn frozen_import_plan_resolves_only_its_imported_prefix() {
        let host = vec![0u8; 128];
        let plan = SessionTransferPlan::new(vec![ImportedHostAllocation {
            base: host.as_ptr() as usize,
            logical_len: host.len(),
            imported_len: 64,
            shards: vec![
                ImportedHostShard {
                    offset: 0,
                    len: 32,
                    buffer: Arc::new(DummyBuffer(32)),
                },
                ImportedHostShard {
                    offset: 32,
                    len: 32,
                    buffer: Arc::new(DummyBuffer(32)),
                },
            ],
        }]);

        let ranges = plan
            .imported_ranges(&host[16..48])
            .expect("range lies in the frozen imported prefix");
        assert_eq!(
            ranges
                .iter()
                .map(|range| (range.offset, range.len))
                .collect::<Vec<_>>(),
            vec![(16, 16), (0, 16)]
        );
        assert!(plan.imported_ranges(&host[64..80]).is_none());

        let unrelated = vec![0u8; 16];
        assert!(plan.imported_ranges(&unrelated).is_none());
    }

    #[test]
    fn shedding_an_idle_import_tail_preserves_the_remaining_prefix() {
        let host = vec![0u8; 64];
        let plan = SessionTransferPlan::new(vec![ImportedHostAllocation {
            base: host.as_ptr() as usize,
            logical_len: host.len(),
            imported_len: host.len(),
            shards: vec![
                ImportedHostShard {
                    offset: 0,
                    len: 32,
                    buffer: Arc::new(DummyBuffer(32)),
                },
                ImportedHostShard {
                    offset: 32,
                    len: 32,
                    buffer: Arc::new(DummyBuffer(32)),
                },
            ],
        }]);

        assert_eq!(plan.shed_unused_import_tails(1), 32);
        assert!(plan.imported_ranges(&host[..32]).is_some());
        assert!(plan.imported_ranges(&host[32..48]).is_none());
    }

    #[test]
    fn shedding_does_not_release_a_tail_retained_by_a_command() {
        let host = vec![0u8; 64];
        let tail: Arc<dyn Buffer> = Arc::new(DummyBuffer(32));
        let command_keepalive = Arc::clone(&tail);
        let plan = SessionTransferPlan::new(vec![ImportedHostAllocation {
            base: host.as_ptr() as usize,
            logical_len: host.len(),
            imported_len: host.len(),
            shards: vec![
                ImportedHostShard {
                    offset: 0,
                    len: 32,
                    buffer: Arc::new(DummyBuffer(32)),
                },
                ImportedHostShard {
                    offset: 32,
                    len: 32,
                    buffer: tail,
                },
            ],
        }]);

        assert_eq!(plan.shed_unused_import_tails(32), 0);
        assert!(plan.imported_ranges(&host[32..]).is_some());
        drop(command_keepalive);
        assert_eq!(plan.shed_unused_import_tails(32), 32);
        assert!(plan.imported_ranges(&host[32..]).is_none());
    }
}
