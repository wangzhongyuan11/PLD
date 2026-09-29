# Package rtl/axis_incr.sv as an IP-repo core (user:pld:axis_incr:1.0) so it
# can be instantiated in a block design and handed to teammate C as a repo
# path. Regenerable at any time from rtl/; ip_repo/ is a build artifact.
#
#   vivado -mode batch -source package_axis_incr.tcl

set repo_root [file normalize [file join [file dirname [info script]] ..]]
set dest "$repo_root/ip_repo/axis_incr"
file delete -force "$dest"

create_project -in_memory -part xc7z020clg400-1
add_files "$repo_root/rtl/axis_incr.sv"
add_files "$repo_root/rtl/axis_reg_slice.sv"
set_property top axis_incr [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir "$dest" -vendor user -library pld \
  -taxonomy /UserIP -import_files -set_current true

set core [ipx::current_core]
set_property display_name {AXIS byte incrementer} $core
set_property description {Placeholder accelerator: byte-wise +1 (mod 256) AXI4-Stream transform, one register stage} $core

# AXI4-Stream slave/master interfaces from port names.
ipx::infer_bus_interfaces xilinx.com:interface:axis_rtl:1.0 $core
set_property interface_mode slave  [ipx::get_bus_interfaces s_axis -of_objects $core]
set_property interface_mode master [ipx::get_bus_interfaces m_axis -of_objects $core]

# Clock association: both streams + reset hang off aclk. These are bus
# parameters (component.xml <spirit:parameter>), not object properties.
set clkif [ipx::get_bus_interfaces aclk -of_objects $core]
if {[ipx::get_bus_parameters ASSOCIATED_BUSIF -of_objects $clkif] eq ""} {
  ipx::add_bus_parameter ASSOCIATED_BUSIF $clkif
}
set_property value {s_axis:m_axis} [ipx::get_bus_parameters ASSOCIATED_BUSIF -of_objects $clkif]
if {[ipx::get_bus_parameters FREQ_HZ -of_objects $clkif] eq ""} {
  ipx::add_bus_parameter FREQ_HZ $clkif
}
set_property value 100000000 [ipx::get_bus_parameters FREQ_HZ -of_objects $clkif]

ipx::save_core
close_project
puts "PLD_INFO: PACKAGE_OK $dest"
