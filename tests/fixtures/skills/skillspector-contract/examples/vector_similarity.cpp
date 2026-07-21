// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <numeric>
#include <vector>

int main() {
  const std::vector<int> values{1, 2, 3};
  return std::accumulate(values.begin(), values.end(), 0) == 6 ? 0 : 1;
}
