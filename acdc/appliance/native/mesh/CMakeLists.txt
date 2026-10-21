cmake_minimum_required(VERSION 3.16)
project(acdc_board_mesh LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(VTK 9.6.0 EXACT REQUIRED COMPONENTS
    CommonCore CommonDataModel CommonTransforms
    FiltersCore FiltersGeneral FiltersModeling IOPLY IOXML)
add_executable(acdc_mesh mesh.cxx)
target_link_libraries(acdc_mesh PRIVATE ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS acdc_mesh MODULES ${VTK_LIBRARIES})
