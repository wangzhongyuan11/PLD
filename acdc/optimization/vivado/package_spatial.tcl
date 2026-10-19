set root [file normalize [file join [file dirname [info script]] ../../..]]
set p [lindex $argv 0]
if {$p ni {4 6}} {error "P must be 4 or 6"}
set dest "$root/ip_repo/acdc_spatial_p$p/spatial_accel"
if {[string first "$root/ip_repo/acdc_spatial_p$p/" $dest] != 0} {error "unsafe generated IP path"}
if {[file exists $dest]} {file delete -force $dest}
create_project -in_memory -part xc7z020clg400-1
add_files "$root/rtl/requant_q24.sv"
foreach n {conv_array conv_compute tile_banks tile_conv stream_fork pool_stream dma_transfer tile_writer parameter_loader input_transform input_loader lut_loader dma_read_arbiter tile_job_queue tile_executor frame_dispatch frame_engine frame_control spatial_accel} {
  add_files "$root/acdc/optimization/rtl/$n.sv"
}
set_property top spatial_accel [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir $dest -vendor user -library pld -taxonomy /UserIP -import_files -set_current true
set core [ipx::current_core]
set_property name spatial_accel $core
set_property display_name {ACD8 autonomous exact INT8 frame accelerator} $core
ipx::infer_bus_interfaces xilinx.com:interface:axis_rtl:1.0 $core
ipx::infer_bus_interfaces xilinx.com:interface:aximm_rtl:1.0 $core
set ctrl [ipx::get_bus_interfaces s_axi -of_objects $core]
set_property interface_mode slave $ctrl
foreach {name value} {PROTOCOL AXI4LITE DATA_WIDTH 32 ADDR_WIDTH 8 READ_WRITE_MODE READ_WRITE} {
  set param [ipx::get_bus_parameters $name -of_objects $ctrl]
  if {$param eq ""} {set param [ipx::add_bus_parameter $name $ctrl]}
  set_property value $value $param
}
set memory [ipx::add_memory_map s_axi $core]
set block [ipx::add_address_block Reg $memory]
set_property -dict {range 256 width 32 usage register base_address 0} $block
set_property slave_memory_map_ref s_axi $ctrl
set clk [ipx::get_bus_interfaces aclk -of_objects $core]
foreach {name value} {ASSOCIATED_BUSIF s_axi:m_axis_read_cmd:s_axis_read_data:s_axis_read_status:m_axis_write_cmd:m_axis_write_data:s_axis_write_status ASSOCIATED_RESET aresetn FREQ_HZ 100000000} {
  set param [ipx::get_bus_parameters $name -of_objects $clk]
  if {$param eq ""} {set param [ipx::add_bus_parameter $name $clk]}
  set_property value $value $param
}
ipx::check_integrity $core
ipx::save_core
close_project
puts "ACDC_SPATIAL_PACKAGE_PASS P=$p path=$dest"
