# Independent Skin8 PS7/DDR/control integration, using the original PS7 preset.
# Usage: vivado -mode batch -source build_board.tcl -tclargs OUT_DIR [setup|bit]
set root [file normalize [file join [file dirname [info script]] ..]]
set repo [file normalize [file join $root .. ..]]
set out [file normalize [lindex $argv 0]]
set mode [lindex $argv 1]
if {$mode ni {setup bit}} {error "mode must be setup or bit"}
file mkdir $out
create_project skin8_board [file join $out project] -part xc7z020clg400-1
add_files [glob [file join $root rtl *.sv]]
update_compile_order -fileset sources_1
source [file join $repo vivado ps7_config.tcl]
create_bd_design skin8_system
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
apply_ps7_config $ps
set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {125} CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_USE_S_AXI_HP2 {0}] $ps
set core [create_bd_cell -type module -reference board_top skin8]
set control [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 control_bus]
set_property CONFIG.NUM_MI 1 $control
set memory [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 memory_bus]
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.ENABLE_ADVANCED_OPTIONS {1} CONFIG.XBAR_DATA_WIDTH {64}] $memory
set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset]
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins control_bus/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control_bus/M00_AXI] [get_bd_intf_pins skin8/S_AXI]
connect_bd_intf_net [get_bd_intf_pins skin8/M_AXI] [get_bd_intf_pins memory_bus/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins memory_bus/M00_AXI] [get_bd_intf_pins ps7/S_AXI_HP0]
make_bd_intf_pins_external [get_bd_intf_pins ps7/DDR]
make_bd_intf_pins_external [get_bd_intf_pins ps7/FIXED_IO]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins skin8/clk] [get_bd_pins reset/slowest_sync_clk] \
  [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins ps7/S_AXI_HP0_ACLK] \
  [get_bd_pins control_bus/ACLK] [get_bd_pins control_bus/S00_ACLK] [get_bd_pins control_bus/M00_ACLK] \
  [get_bd_pins memory_bus/ACLK] [get_bd_pins memory_bus/S00_ACLK] [get_bd_pins memory_bus/M00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins reset/ext_reset_in]
connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins skin8/resetn] \
  [get_bd_pins control_bus/ARESETN] [get_bd_pins control_bus/S00_ARESETN] [get_bd_pins control_bus/M00_ARESETN] \
  [get_bd_pins memory_bus/ARESETN] [get_bd_pins memory_bus/S00_ARESETN] [get_bd_pins memory_bus/M00_ARESETN]
assign_bd_address -offset 0x43C00000 -range 0x1000 -target_address_space [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs -of_objects [get_bd_intf_pins skin8/S_AXI]]
assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces -of_objects [get_bd_intf_pins skin8/M_AXI]] [get_bd_addr_segs ps7/S_AXI_HP0/HP0_DDR_LOWOCM]
validate_bd_design
save_bd_design
write_bd_tcl -force [file join $out system_bd.tcl]
if {$mode eq "setup"} {
  puts "SKIN8_BD_SETUP_PASS"
  close_project
  exit
}
generate_target all [get_files skin8_system.bd]
make_wrapper -files [get_files skin8_system.bd] -top -import
set_property top skin8_system_wrapper [current_fileset]
update_compile_order -fileset sources_1
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "implementation failed"}
open_run impl_1
report_timing_summary -file [file join $out timing.rpt]
report_utilization -file [file join $out utilization.rpt]
report_drc -file [file join $out drc.rpt]
set setup_slack [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold_slack [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
if {$setup_slack < 0 || $hold_slack < 0} {error "negative setup/hold slack"}
write_hw_platform -fixed -include_bit -force [file join $out skin8.xsa]
puts "SKIN8_FULL_BUILD_PASS setup=$setup_slack hold=$hold_slack"
close_project
