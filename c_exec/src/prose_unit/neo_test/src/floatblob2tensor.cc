//
// Created by Entropy Xu on 9/10/24.
//

#include <fstream>
#include <util.h>

#ifdef USE_TORCH

#include "torch/torch.h"
#include "interchange_format.h"

int main(int argc, char ** argv) {
  auto q = interchange_format::from_float_file(argv[1], {1, 4, 768});
  auto t = torch::from_blob(q.data, {1, 4, 768});
  auto r = torch::pickle_save(t);
  FILE *f = fopen(argv[2], "w");
  for (const char &c: r) {
    fputc(c, f);
  }
  fclose(f);
  return 0;
}
#else

int main() {}

#endif