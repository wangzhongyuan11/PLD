set repo_root [file normalize [file join [file dirname [info script]] ../..]]
set dest [file normalize "$repo_root/ip_repo/camus/axis_tensor_accel_camus"]
if {[string first "$repo_root/ip_repo/" $dest] != 0 || [file tail $dest] ne "axis_tensor_accel_camus"} {
  error "Refusing to replace IP outside the generated repository"
}
file delete -force $dest
create_project -in_memory -part xc7z020clg400-1
foreach module {conv_mac_array requant_q24 axi_lite_endpoint axis_byte_pack axis_byte_unpack} {
  add_files "$repo_root/rtl/${module}.sv"
}
foreach module {conv_tile_engine_camus tensor_tile_engine_camus axis_tensor_accel_camus} {
  add_files "$repo_root/camus/rtl/${module}.sv"
}
set_property top axis_tensor_accel_camus [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir $dest -vendor user -library pld -taxonomy /UserIP -import_files -set_current true
set core [ipx::current_core]
set_property display_name {CAMUS CAM2 INT8 accelerator} $core
set_property description {CAM2 ABI 2 banked convolution and tensor operators with bounded software-controlled commands} $core
ipx::infer_bus_interfaces xilinx.com:interface:axis_rtl:1.0 $core
ipx::infer_bus_interfaces xilinx.com:interface:aximm_rtl:1.0 $core
set_property interface_mode slave [ipx::get_bus_interfaces s_axis -of_objects $core]
set_property interface_mode master [ipx::get_bus_interfaces m_axis -of_objects $core]
set ctrl [ipx::get_bus_interfaces s_axi -of_objects $core]
set_property interface_mode slave $ctrl
foreach {name value} {PROTOCOL AXI4LITE DATA_WIDTH 32 ADDR_WIDTH 12 READ_WRITE_MODE READ_WRITE} {
  set parameter [ipx::get_bus_parameters $name -of_objects $ctrl]
  if {$parameter eq ""} {set parameter [ipx::add_bus_parameter $name $ctrl]}
  set_property value $value $parameter
}
set memory [ipx::get_memory_maps s_axi -of_objects $core]
if {$memory eq ""} {set memory [ipx::add_memory_map s_axi $core]}
set block [ipx::get_address_blocks Reg -of_objects $memory]
if {$block eq ""} {set block [ipx::add_address_block Reg $memory]}
set_property range 4096 $block
set_property width 32 $block
set_property usage register $block
set_property base_address 0 $block
set_property slave_memory_map_ref s_axi $ctrl
set clkif [ipx::get_bus_interfaces aclk -of_objects $core]
# PS7 2026.1 propagates this integer for the requested 1000/8 MHz clock.
foreach {name value} {ASSOCIATED_BUSIF s_axis:m_axis:s_axi ASSOCIATED_RESET aresetn FREQ_HZ 100000000} {
  set parameter [ipx::get_bus_parameters $name -of_objects $clkif]
  if {$parameter eq ""} {set parameter [ipx::add_bus_parameter $name $clkif]}
  set_property value $value $parameter
}
ipx::check_integrity $core
ipx::save_core
close_project
puts "PLD_INFO: PACKAGE_OK $dest"
