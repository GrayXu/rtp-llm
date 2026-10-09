#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"

#include <exception>
#include <cerrno>
#include <cstring>
#include <limits>

#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/memfd.h>
#include <sys/syscall.h>
#endif

#if USING_CUDA
#include <cuda_runtime.h>
#endif

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

AlignedHostMemory::AlignedHostMemory(size_t             usable_bytes,
                                     size_t             alignment,
                                     const std::string& allocation_name,
                                     bool               shared_memory): size_(usable_bytes) {
    RTP_LLM_CHECK_WITH_INFO(usable_bytes > 0 && alignment > 0, "invalid host allocation size or alignment");
    if (shared_memory) {
#if USING_CUDA && defined(__linux__)
        const long page_size = sysconf(_SC_PAGESIZE);
        RTP_LLM_CHECK_WITH_INFO(page_size > 0 && static_cast<size_t>(page_size) % alignment == 0
                                    && usable_bytes <= static_cast<size_t>(std::numeric_limits<off_t>::max()),
                                "shared host allocation requires page-compatible alignment and size");
        fd_ = static_cast<int>(syscall(SYS_memfd_create, allocation_name.c_str(), MFD_CLOEXEC));
        RTP_LLM_CHECK_WITH_INFO(fd_ >= 0, "create shared host backing failed: %s", std::strerror(errno));
        if (ftruncate(fd_, static_cast<off_t>(usable_bytes)) != 0) {
            const int error = errno;
            close(fd_);
            fd_ = -1;
            RTP_LLM_FAIL("size shared host backing failed: %s", std::strerror(error));
        }
        void* mapped = mmap(nullptr, usable_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED) {
            const int error = errno;
            close(fd_);
            fd_ = -1;
            RTP_LLM_FAIL("map shared host backing failed: %s", std::strerror(error));
        }
        if (cudaHostRegister(mapped, usable_bytes, cudaHostRegisterDefault) != cudaSuccess) {
            munmap(mapped, usable_bytes);
            close(fd_);
            fd_ = -1;
            RTP_LLM_FAIL("pin shared host backing failed, allocation=%s", allocation_name.c_str());
        }
        data_ = static_cast<uint8_t*>(mapped);
        return;
#else
        RTP_LLM_FAIL("shared pinned host backing requires CUDA");
#endif
    }
    RTP_LLM_CHECK_WITH_INFO(alignment <= static_cast<size_t>(std::numeric_limits<int64_t>::max())
                                && usable_bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) - alignment,
                            "pinned host allocation size overflow");
    try {
        backing_ = torch::empty({static_cast<int64_t>(usable_bytes + alignment)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("allocate pinned host memory failed, allocation=%s usable_bytes=%zu error=%s",
                     allocation_name.c_str(),
                     usable_bytes,
                     e.what());
    }
    RTP_LLM_CHECK_WITH_INFO(
        backing_.is_pinned(), "host allocation [%s] must use pinned CPU memory", allocation_name.c_str());

    const auto raw_base = reinterpret_cast<uintptr_t>(backing_.data_ptr<uint8_t>());
    data_               = reinterpret_cast<uint8_t*>((raw_base + alignment - 1) / alignment * alignment);
}

AlignedHostMemory::~AlignedHostMemory() {
    if (fd_ >= 0) {
#if USING_CUDA
        cudaHostUnregister(data_);
#endif
        munmap(data_, size_);
        close(fd_);
    }
}

uint8_t* AlignedHostMemory::data() const {
    return data_;
}

int AlignedHostMemory::fd() const {
    return fd_;
}

size_t AlignedHostMemory::size() const {
    return size_;
}

void AlignedHostMemory::preserveUntilProcessExit() {
    // An SDK timeout may return before a lower-level task has stopped using
    // this mapping. Keep the pinned mapping and fd alive until process exit.
    fd_ = -1;
}

}  // namespace rtp_llm
