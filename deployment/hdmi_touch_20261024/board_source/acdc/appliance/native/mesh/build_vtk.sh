#!/usr/bin/env bash
# Build an isolated, headless ARM VTK. No system package/venv is modified.
set -euo pipefail
work="${1:?absolute dedicated build directory required}"
archive="${2:?VTK-9.6.0 source tar.gz required}"
mkdir -p "$work"
if [[ ! -d "$work/VTK-9.6.0" ]]; then
    tar -mxzf "$archive" -C "$work"
fi
cmake -S "$work/VTK-9.6.0" -B "$work/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX="$work/install" \
    -DCMAKE_C_FLAGS='-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard' \
    -DCMAKE_CXX_FLAGS='-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard' \
    -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF -DVTK_BUILD_DOCUMENTATION=OFF \
    -DVTK_WRAP_PYTHON=OFF -DVTK_WRAP_JAVA=OFF \
    -DVTK_GROUP_ENABLE_StandAlone=DONT_WANT -DVTK_GROUP_ENABLE_Rendering=DONT_WANT \
    -DVTK_GROUP_ENABLE_Imaging=DONT_WANT -DVTK_GROUP_ENABLE_Views=DONT_WANT \
    -DVTK_MODULE_ENABLE_VTK_FiltersCore=YES \
    -DVTK_MODULE_ENABLE_VTK_FiltersGeneral=YES \
    -DVTK_MODULE_ENABLE_VTK_FiltersModeling=YES \
    -DVTK_MODULE_ENABLE_VTK_IOPLY=YES -DVTK_MODULE_ENABLE_VTK_IOXML=YES \
    -DVTK_SMP_IMPLEMENTATION_TYPE=STDThread
cmake --build "$work/build" --parallel 1
cmake --install "$work/build"
