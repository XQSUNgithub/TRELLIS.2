#!/usr/bin/env bash
set -euo pipefail

# Clone external repositories into project root.
# Run this script in an environment that can access github.com.

git clone https://github.com/Nelipot-Lee/VoxHammer.git voxhammer
git clone https://github.com/wangjiangshan0725/RF-Solver-Edit.git RF-Solver-Edit
