# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import vllm

from .mccl import find_mccl_library

import vllm.utils.nccl

vllm.utils.nccl.find_nccl_library = find_mccl_library
