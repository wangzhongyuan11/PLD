#!/usr/bin/env bash
# Linux/WSL host. Requires GCC 11 armhf, CMake, Ninja, native C/C++ compilers.
set -euo pipefail
work="${1:?new dedicated absolute build directory}"
archive="${2:?official VTK-9.6.0 tar.gz}"
sdk="${3:?captured board sysroot tar.gz}"
here="$(cd "$(dirname "$0")" && pwd)"
[[ "$work" = /* && ! -e "$work" ]] || { echo 'choose a new absolute build directory' >&2; exit 1; }
echo 'd77d180694faafdc816578b9a53651f6790e799615811bfbb91018661a3bb8f2  '"$archive" | sha256sum -c -
[[ "${ACDC_LARGE_FILE_PROBE_EXITCODE:-}" = 0 ]] || { echo 'run the VTK large-file probe on this board first; set ACDC_LARGE_FILE_PROBE_EXITCODE=0 only after success' >&2; exit 1; }
mkdir -p "$work/sysroot"
tar -mxzf "$archive" -C "$work"
tar -mxzf "$sdk" -C "$work/sysroot"
cmake -G Ninja -S "$work/VTK-9.6.0" -B "$work/host" \
    -DCMAKE_BUILD_TYPE=Release -DVTK_BUILD_COMPILE_TOOLS_ONLY=ON \
    -DVTK_BUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build "$work/host" --parallel "${ACDC_BUILD_JOBS:-4}"
host="$work/host/lib/cmake/vtkcompiletools-9.6"
# These executables run on the 64-bit host. Bypass only their package's pointer
# size suitability check, which incorrectly compares host tools to ARM32.
printf '\nset(PACKAGE_VERSION_UNSUITABLE FALSE)\n' >> "$host/vtkcompiletools-config-version.cmake"
export ACDC_BOARD_SYSROOT="$work/sysroot"
cmake -G Ninja -S "$work/VTK-9.6.0" -B "$work/arm-board" \
    -DCMAKE_TOOLCHAIN_FILE="$here/arm-board-toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DVTKCompileTools_DIR="$host" -DVTK_REQUIRE_LARGE_FILE_SUPPORT_EXITCODE=0 \
    -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF -DVTK_BUILD_DOCUMENTATION=OFF \
    -DVTK_WRAP_PYTHON=OFF -DVTK_WRAP_JAVA=OFF -DVTK_USE_64BIT_IDS=OFF -DVTK_LEGACY_REMOVE=ON \
    -DVTK_GROUP_ENABLE_StandAlone=DONT_WANT -DVTK_GROUP_ENABLE_Rendering=DONT_WANT \
    -DVTK_GROUP_ENABLE_Imaging=DONT_WANT -DVTK_GROUP_ENABLE_Views=DONT_WANT \
    -DVTK_MODULE_ENABLE_VTK_FiltersCore=YES -DVTK_MODULE_ENABLE_VTK_FiltersGeneral=YES \
    -DVTK_MODULE_ENABLE_VTK_FiltersModeling=YES -DVTK_MODULE_ENABLE_VTK_IOPLY=YES \
    -DVTK_MODULE_ENABLE_VTK_IOXML=YES -DVTK_SMP_IMPLEMENTATION_TYPE=STDThread
cmake --build "$work/arm-board" --parallel "${ACDC_BUILD_JOBS:-4}"
cmake -G Ninja -S "$here" -B "$work/mesh-board" \
    -DCMAKE_TOOLCHAIN_FILE="$here/arm-board-toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DVTK_DIR="$work/arm-board/lib/cmake/vtk-9.6"
cmake --build "$work/mesh-board" --parallel 2
sha256sum "$sdk" "$work/mesh-board/acdc_mesh"
arm-linux-gnueabihf-readelf -W -V "$work/mesh-board/acdc_mesh" > "$work/mesh-board/elf-versions.txt"
echo 'Run acdc_mesh --version and authenticated label/geometry comparisons on the board before deployment.'
