#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <torch/torch.h>

namespace rtp_llm {

class AlignedHostMemory {
public:
    AlignedHostMemory(size_t usable_bytes,
                      size_t alignment,
                      const std::string& allocation_name,
                      bool shared_memory = false);
    ~AlignedHostMemory();

    AlignedHostMemory(const AlignedHostMemory&) = delete;
    AlignedHostMemory& operator=(const AlignedHostMemory&) = delete;

    uint8_t* data() const;
    int      fd() const;
    size_t   size() const;
    void     preserveUntilProcessExit();

private:
    torch::Tensor backing_;
    uint8_t*      data_{nullptr};
    size_t        size_{0};
    int           fd_{-1};
};

}  // namespace rtp_llm
