# Independent PS7 / DataMover Full / ACD8 SoC. GPIO control holds the DMA/core
# in reset at power-up; software must explicitly release it. HALT completion
# is observable before reset or CMA reuse. No Skin/CAMUS files are generated.
set root [file normalize [file join [file dirname [info script]] ../../..]]
set p [lindex $argv 0]
if {$p ni {4 6}} {error "P must be 4 or 6"}
set tag [lindex $argv 1]
if {$tag ne "" && ![regexp {^[a-zA-Z0-9_-]+$} $tag]} {error "invalid build tag"}
set suffix [expr {$tag eq "" ? "" : "_$tag"}]
set out "$root/out/acdc/optimization/spatial_p${p}_100$suffix"
file mkdir $out
exec "$root/acdc/.venv/Scripts/python.exe" -m acdc.optimization.package $out --p $p --begin
create_project -force spatial "$out/project" -part xc7z020clg400-1
set_property ip_repo_paths [list "$root/ip_repo/acdc_spatial_p$p"] [current_project]
update_ip_catalog
source "$root/vivado/ps7_config.tcl"
create_bd_design dma_top
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
apply_ps7_config $ps
set_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100 $ps
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset]
set core [create_bd_cell -type ip -vlnv user:pld:spatial_accel:1.0 spatial_accel_0]
set_property -dict [list CONFIG.P $p CONFIG.CLOCK_HZ 100000000] $core
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_datamover:5.1 datamover]
set_property -dict {
  CONFIG.c_include_mm2s Full CONFIG.c_include_s2mm Full
  CONFIG.c_m_axi_mm2s_data_width 64 CONFIG.c_m_axis_mm2s_tdata_width 64
  CONFIG.c_m_axi_s2mm_data_width 64 CONFIG.c_s_axis_s2mm_tdata_width 64
  CONFIG.c_addr_width 32 CONFIG.c_include_mm2s_dre 0 CONFIG.c_include_s2mm_dre 0
  CONFIG.c_mm2s_burst_size 16 CONFIG.c_s2mm_burst_size 16
  CONFIG.c_mm2s_btt_used 23 CONFIG.c_s2mm_btt_used 23
  CONFIG.c_enable_mm2s_adv_sig 1 CONFIG.c_enable_s2mm_adv_sig 1
} $dma
set ctl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 control]
set_property CONFIG.NUM_MI 2 $ctl
foreach {name hp} {read_path HP2 write_path HP0} {
  set cell [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 $name]
  set_property -dict {CONFIG.NUM_MI 1 CONFIG.XBAR_DATA_WIDTH 64} $cell
}
set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 dma_control]
set_property -dict {CONFIG.C_IS_DUAL 1 CONFIG.C_GPIO_WIDTH 2 CONFIG.C_ALL_OUTPUTS 1 CONFIG.C_DOUT_DEFAULT 0x00000000 CONFIG.C_GPIO2_WIDTH 4 CONFIG.C_ALL_INPUTS_2 1} $gpio
foreach {name from to} {enable 0 0 halt 1 1} {
  set c [create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 $name]
  set_property -dict [list CONFIG.DIN_WIDTH 2 CONFIG.DIN_FROM $from CONFIG.DIN_TO $to CONFIG.DOUT_WIDTH 1] $c
  connect_bd_net [get_bd_pins dma_control/gpio_io_o] [get_bd_pins $name/Din]
}
set gate [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 run_reset]
set_property -dict {CONFIG.C_SIZE 1 CONFIG.C_OPERATION and} $gate
connect_bd_net [get_bd_pins enable/Dout] [get_bd_pins run_reset/Op1]
connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins run_reset/Op2]
set status [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 dma_status]
set_property CONFIG.NUM_PORTS 4 $status
foreach {port pin} {In0 mm2s_halt_cmplt In1 s2mm_halt_cmplt In2 mm2s_err In3 s2mm_err} {
  connect_bd_net [get_bd_pins datamover/$pin] [get_bd_pins dma_status/$port]
}
connect_bd_net [get_bd_pins dma_status/dout] [get_bd_pins dma_control/gpio2_io_i]
connect_bd_net [get_bd_pins halt/Dout] [get_bd_pins datamover/mm2s_halt] [get_bd_pins datamover/s2mm_halt]
set one [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 allow_address]
set_property -dict {CONFIG.CONST_WIDTH 1 CONFIG.CONST_VAL 1} $one
connect_bd_net [get_bd_pins allow_address/dout] [get_bd_pins datamover/mm2s_allow_addr_req] [get_bd_pins datamover/s2mm_allow_addr_req]
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins control/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control/M00_AXI] [get_bd_intf_pins spatial_accel_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins control/M01_AXI] [get_bd_intf_pins dma_control/S_AXI]
foreach {dm ci} {M_AXI_MM2S read_path M_AXI_S2MM write_path} {
  connect_bd_intf_net [get_bd_intf_pins datamover/$dm] [get_bd_intf_pins $ci/S00_AXI]
}
connect_bd_intf_net [get_bd_intf_pins read_path/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP2]
connect_bd_intf_net [get_bd_intf_pins write_path/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
foreach {a b} {m_axis_read_cmd S_AXIS_MM2S_CMD s_axis_read_data M_AXIS_MM2S s_axis_read_status M_AXIS_MM2S_STS m_axis_write_cmd S_AXIS_S2MM_CMD m_axis_write_data S_AXIS_S2MM s_axis_write_status M_AXIS_S2MM_STS} {
  connect_bd_intf_net [get_bd_intf_pins spatial_accel_0/$a] [get_bd_intf_pins datamover/$b]
}
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR
connect_bd_intf_net [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO
connect_bd_intf_net [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
set clk [get_bd_pins processing_system7_0/FCLK_CLK0]
foreach name {processing_system7_0/M_AXI_GP0_ACLK processing_system7_0/S_AXI_HP0_ACLK processing_system7_0/S_AXI_HP2_ACLK reset/slowest_sync_clk spatial_accel_0/aclk dma_control/s_axi_aclk datamover/m_axi_mm2s_aclk datamover/m_axi_s2mm_aclk datamover/m_axis_mm2s_cmdsts_aclk datamover/m_axis_s2mm_cmdsts_awclk} {
  connect_bd_net $clk [get_bd_pins $name]
}
foreach ci {control read_path write_path} {
  foreach pin {ACLK S00_ACLK M00_ACLK} {connect_bd_net $clk [get_bd_pins $ci/$pin]}
  foreach pin {ARESETN S00_ARESETN M00_ARESETN} {connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins $ci/$pin]}
}
connect_bd_net $clk [get_bd_pins control/M01_ACLK]
connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins control/M01_ARESETN] [get_bd_pins dma_control/s_axi_aresetn]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins reset/ext_reset_in]
foreach name {spatial_accel_0/aresetn datamover/m_axi_mm2s_aresetn datamover/m_axi_s2mm_aresetn datamover/m_axis_mm2s_cmdsts_aresetn datamover/m_axis_s2mm_cmdsts_aresetn} {
  connect_bd_net [get_bd_pins run_reset/Res] [get_bd_pins $name]
}
assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces datamover/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP2/HP2_DDR_LOWOCM]
assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces datamover/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM]
assign_bd_address -offset 0x43c00000 -range 0x00000100 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs spatial_accel_0/s_axi/Reg]
assign_bd_address -offset 0x41200000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs dma_control/S_AXI/Reg]
validate_bd_design
save_bd_design
generate_target all [get_files dma_top.bd]
make_wrapper -files [get_files dma_top.bd] -top -import
set_property top dma_top_wrapper [current_fileset]
update_compile_order -fileset sources_1
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "spatial implementation incomplete"}
open_run impl_1
report_timing_summary -file "$out/timing_summary.rpt"
report_utilization -file "$out/utilization.rpt"
report_drc -file "$out/drc.rpt"
write_hw_platform -fixed -include_bit -force "$out/dma_top.xsa"
puts "ACDC_SPATIAL_BUILD_DONE P=$p output=$out check_routed_timing_before_use"
