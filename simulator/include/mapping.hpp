#ifndef SMARTGC_MAPPING_HPP
#define SMARTGC_MAPPING_HPP

#include "types.hpp"
#include <unordered_map>

namespace smartgc {

class LbaMapping {
public:
    LbaMapping() = default;

    bool contains(LbaType lba) const {
        return table_.find(lba) != table_.end();
    }

    bool get(LbaType lba, PhysicalAddress& out_addr) const {
        auto it = table_.find(lba);
        if (it != table_.end()) {
            out_addr = it->second;
            return true;
        }
        return false;
    }

    const PhysicalAddress* find(LbaType lba) const {
        auto it = table_.find(lba);
        if (it != table_.end()) {
            return &(it->second);
        }
        return nullptr;
    }

    void set(LbaType lba, PhysicalAddress addr) {
        table_[lba] = addr;
    }

    bool remove(LbaType lba) {
        return table_.erase(lba) > 0;
    }

    size_t size() const {
        return table_.size();
    }

    void clear() {
        table_.clear();
    }

    const std::unordered_map<LbaType, PhysicalAddress>& table() const {
        return table_;
    }

private:
    std::unordered_map<LbaType, PhysicalAddress> table_;
};

} // namespace smartgc

#endif // SMARTGC_MAPPING_HPP
