// A bespoke Omniprobe message handler: counts, per kernel dispatch, the address
// messages (one per wave execution of an instrumented memory instruction) by
// memory space and access kind, the lanes that took part, and the executions
// by partially active waves (fewer than 64 lanes). Nothing in stock Omniprobe
// reports these counts.
//
// Build (flags from `omniprobe-ik handler-flags`):
//   hipcc -std=c++20 -fPIC -shared -O2 -I<omniprobe>/include/dh_comms access_counter.cc \
//         -o libaccess_counter.so -L<omniprobe>/lib -ldh_comms -Wl,-rpath,<omniprobe>/lib
// Run (the application built with `omniprobe-ik flags memory`):
//   omniprobe -i -a /abs/path/libaccess_counter.so -- ./app
//   omniprobe-ik run /abs/path/libaccess_counter.so ./app
//
// Contract: Omniprobe calls getMessageHandlers() once per kernel dispatch; each
// handler's handle() sees every message from the device and returns true if it
// consumed it; report() runs when the dispatch's messages are all processed.
// Address messages (user_type == address) carry one 64-bit address per active
// lane; user_data packs the access kind (bits 0-1: 1 read, 2 write), the address
// space (bits 2-5) and the access size in bytes (bits 6+).
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "dh_comms.h"
#include "message_handlers.h"

namespace {

const char *space_name(unsigned space) {
  switch (space) {
  case dh_comms::address_space::global:
    return "global";
  case dh_comms::address_space::shared:
    return "lds";
  case dh_comms::address_space::constant:
    return "constant";  // e.g. kernel-argument loads
  case dh_comms::address_space::flat:
    return "flat";
  case dh_comms::address_space::scratch:
    return "scratch";
  default:
    return "other";
  }
}

class access_counter_t : public dh_comms::message_handler_base {
public:
  access_counter_t(std::string kernel, uint64_t dispatch_id)
      : kernel_(std::move(kernel)), dispatch_id_(dispatch_id) {}

  bool handle(const dh_comms::message_t &message) override {
    const auto &header = message.wave_header();
    if (header.user_type != dh_comms::message_type::address) {
      return false;  // not ours: let other handlers see it
    }
    const unsigned rw = header.user_data & 0x3;
    const unsigned space = (header.user_data >> 2) & 0xf;
    const size_t lanes = message.no_data_items();
    Counts &c = counts_[space_name(space)];
    (rw == dh_comms::memory_access::write ? c.store_executions : c.load_executions) += 1;
    c.lanes += lanes;
    if (lanes < 64) {
      c.partial_wave_executions += 1;
    }
    return true;
  }

  void report() override {
    for (const auto &[space, c] : counts_) {
      std::printf("ACCESS_COUNTER {\"kernel\": \"%s\", \"dispatch\": %lu, \"space\": \"%s\", "
                  "\"load_executions\": %lu, \"store_executions\": %lu, \"lanes\": %lu, "
                  "\"partial_wave_executions\": %lu}\n",
                  kernel_.c_str(), (unsigned long)dispatch_id_, space.c_str(),
                  c.load_executions, c.store_executions, c.lanes, c.partial_wave_executions);
    }
    std::fflush(stdout);
  }

  void clear() override { counts_.clear(); }

private:
  struct Counts {
    unsigned long load_executions = 0, store_executions = 0, lanes = 0, partial_wave_executions = 0;
  };
  std::string kernel_;
  uint64_t dispatch_id_;
  std::map<std::string, Counts> counts_;
};

}  // namespace

extern "C" {
__attribute__((visibility("default"))) void getMessageHandlers(const std::string &kernel, uint64_t dispatch_id,
                                   std::vector<dh_comms::message_handler_base *> &outHandlers) {
  outHandlers.push_back(new access_counter_t(kernel, dispatch_id));
}
}
