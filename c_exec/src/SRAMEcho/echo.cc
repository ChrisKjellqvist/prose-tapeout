//
// Created by Christopher Kjellqvist on 9/6/24.
//

#include <random>

#include <beethoven/fpga_handle.h>
#include <beethoven_allocator_declaration.h>
#include <cinttypes>

int main() {
  // generate a random 32-bit int
  std::random_device rd;
  // random address is 8-bits
  std::uniform_int_distribution<uint32_t> dist(0, 255);
  // random payload is 32-bits
  std::uniform_int_distribution<uint32_t> dist_payload(0, 4294967295);
  std::default_random_engine eng(rd());

  auto addr = dist(eng);
  auto payload = dist_payload(eng);

  beethoven::fpga_handle_t handle;

  SRAMEcho::write(0, addr, payload);
  std::cout << "enqueued\n" << std::endl;

  // make sure it echos back the same value
  auto resp = SRAMEcho::read(0, addr).get();

  std::cout << "Wrote " << payload << " to address " << addr << " and read back " << resp.payload << std::endl;
  if (resp.payload != payload) {
    std::cout << "ERROR: read back value does not match written value" << std::endl;
    return 1;
  } else {
    std::cout << "SUCCESS: read back value matches written value" << std::endl;
    return 0;
  }


}