# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

CODES = {"cobalt": "ZX-731", "amber": "QW-482"}
print(CODES[sys.argv[1]])
