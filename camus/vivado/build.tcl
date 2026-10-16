# Build the DMA loopback overlay (stage 1/2) in Vivado batch mode.
#
#   vivado -mode batch -source build_dma_bd.tcl -tclargs <pl_core> <out_root>
#
#   pl_core = fifo : BD loopback via axis_data_fifo (stage 1, replica of the
#                     PYNQ_tutorials/dma design, rebuilt with Vivado 2026.1)
#   pl_core = incr : loopback via the self-written axis_incr IP (stage 2)
#
# Produces <out_root>/dma_top.xsa (bit+hwh inside) plus timing/utilization
# reports; bit/hwh are extracted to <out_root>/ by scripts/extract_xsa.py.

set pl_core [lindex $argv 0]
if {$pl_core ne "fifo" && $pl_core ne "incr" && $pl_core ne "cnn"} {
  error "usage: -tclargs <fifo|incr|cnn> <out_root>"
}
set out_root [file normalize [lindex $argv 1]]
set repo_root [file normalize [file join [file dirname [info script]] ../..]]

file mkdir $out_root
create_project -force dma_proj "$out_root/proj" -part xc7z020clg400-1

if {$pl_core eq "incr" || $pl_core eq "cnn"} {
  set_property ip_repo_paths [list "$repo_root/ip_repo/camus"] [current_project]
  update_ip_catalog
}

source [file join $repo_root vivado ps7_config.tcl]

create_bd_design dma_top

# --- cells -----------------------------------------------------------------
set axi_dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma]
set_property -dict [list \
  CONFIG.c_include_sg {0} \
  CONFIG.c_m_axi_mm2s_data_width {64} \
  CONFIG.c_m_axis_mm2s_tdata_width {64} \
  CONFIG.c_mm2s_burst_size {8} \
  CONFIG.c_sg_include_stscntrl_strm {0} \
  CONFIG.c_sg_length_width {26} \
] $axi_dma

set axi_mem_intercon   [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon]
set_property -dict [list CONFIG.ENABLE_ADVANCED_OPTIONS {1} CONFIG.NUM_MI {1} CONFIG.XBAR_DATA_WIDTH {64}] $axi_mem_intercon

set axi_mem_intercon_1 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon_1]
set_property -dict [list CONFIG.ENABLE_ADVANCED_OPTIONS {1} CONFIG.NUM_MI {1} CONFIG.XBAR_DATA_WIDTH {64}] $axi_mem_intercon_1

set ps7_0_axi_periph   [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph]
set_property -dict [list CONFIG.NUM_MI {1}] $ps7_0_axi_periph
if {$pl_core eq "cnn"} {set_property CONFIG.NUM_MI 2 $ps7_0_axi_periph}

set rst_ps7_0_100M     [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M]

set processing_system7_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
apply_ps7_config $processing_system7_0
if {$pl_core eq "cnn"} {
  # 1000 MHz IO PLL / 8. Keep the seed and historical FIFO/INCR clocks intact.
  set_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} $processing_system7_0
}

# --- PL loopback core (the only stage-dependent cell) -----------------------
set clk_pins  [list]
set rst_pins  [list]
if {$pl_core eq "fifo"} {
  set lb [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_loopback]
  lappend intf_out [get_bd_intf_pins axis_data_fifo_loopback/S_AXIS]
  lappend intf_in  [get_bd_intf_pins axis_data_fifo_loopback/M_AXIS]
  lappend clk_pins [get_bd_pins axis_data_fifo_loopback/s_axis_aclk]
  lappend rst_pins [get_bd_pins axis_data_fifo_loopback/s_axis_aresetn]
} elseif {$pl_core eq "incr"} {
  set lb [create_bd_cell -type ip -vlnv user:pld:axis_incr:1.0 axis_incr_0]
  set_property -dict [list CONFIG.DATA_W {64}] $lb
  lappend intf_out [get_bd_intf_pins axis_incr_0/s_axis]
  lappend intf_in  [get_bd_intf_pins axis_incr_0/m_axis]
  lappend clk_pins [get_bd_pins axis_incr_0/aclk]
  lappend rst_pins [get_bd_pins axis_incr_0/aresetn]
} else {
  set lb [create_bd_cell -type ip -vlnv user:pld:axis_tensor_accel_camus:1.0 axis_tensor_accel_camus_0]
  lappend intf_out [get_bd_intf_pins axis_tensor_accel_camus_0/s_axis]
  lappend intf_in [get_bd_intf_pins axis_tensor_accel_camus_0/m_axis]
  lappend clk_pins [get_bd_pins axis_tensor_accel_camus_0/aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK]
  lappend rst_pins [get_bd_pins axis_tensor_accel_camus_0/aresetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN]
  connect_bd_intf_net [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axis_tensor_accel_camus_0/s_axi]
}

# --- interface connections --------------------------------------------------
connect_bd_intf_net -intf_net axi_dma_M_AXIS_MM2S [get_bd_intf_pins axi_dma/M_AXIS_MM2S] $intf_out
connect_bd_intf_net -intf_net pl_loopback_M_AXIS  $intf_in [get_bd_intf_pins axi_dma/S_AXIS_S2MM]
connect_bd_intf_net -intf_net axi_dma_M_AXI_MM2S [get_bd_intf_pins axi_dma/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon_1/S00_AXI]
connect_bd_intf_net -intf_net axi_dma_M_AXI_S2MM [get_bd_intf_pins axi_dma/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
connect_bd_intf_net -intf_net axi_mem_intercon_1_M00_AXI [get_bd_intf_pins axi_mem_intercon_1/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP2]
connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR
connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO
connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins axi_dma/S_AXI_LITE] [get_bd_intf_pins ps7_0_axi_periph/M00_AXI]

# --- clock/reset nets -------------------------------------------------------
connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] \
  [get_bd_pins axi_dma/m_axi_mm2s_aclk] [get_bd_pins axi_dma/m_axi_s2mm_aclk] [get_bd_pins axi_dma/s_axi_lite_aclk] \
  [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/S00_ACLK] \
  [get_bd_pins axi_mem_intercon_1/ACLK] [get_bd_pins axi_mem_intercon_1/M00_ACLK] [get_bd_pins axi_mem_intercon_1/S00_ACLK] \
  {*}$clk_pins \
  [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP2_ACLK] \
  [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] \
  [get_bd_pins rst_ps7_0_100M/slowest_sync_clk]

connect_bd_net -net processing_system7_0_FCLK_RESET0_N \
  [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_100M/ext_reset_in]

connect_bd_net -net rst_ps7_0_100M_peripheral_aresetn \
  [get_bd_pins axi_dma/axi_resetn] \
  [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] \
  [get_bd_pins axi_mem_intercon_1/ARESETN] [get_bd_pins axi_mem_intercon_1/M00_ARESETN] [get_bd_pins axi_mem_intercon_1/S00_ARESETN] \
  {*}$rst_pins \
  [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] \
  [get_bd_pins rst_ps7_0_100M/peripheral_aresetn]

# --- address map (same as seed) ----------------------------------------------
assign_bd_address -offset 0x00000000 -range 0x20000000 \
  -target_address_space [get_bd_addr_spaces axi_dma/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM]
assign_bd_address -offset 0x00000000 -range 0x20000000 \
  -target_address_space [get_bd_addr_spaces axi_dma/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP2/HP2_DDR_LOWOCM]
assign_bd_address -offset 0x40400000 -range 0x00010000 \
  -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma/S_AXI_LITE/Reg]
if {$pl_core eq "cnn"} {
  assign_bd_address -offset 0x43C00000 -range 0x00001000 \
    -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axis_tensor_accel_camus_0/s_axi/Reg]
}

validate_bd_design
save_bd_design

# --- synth/impl/bitstream ----------------------------------------------------
generate_target all [get_files dma_top.bd]
make_wrapper -files [get_files dma_top.bd] -top -import
set_property top dma_top_wrapper [current_fileset]
update_compile_order -fileset sources_1

if {$pl_core eq "cnn"} {
  # Pre-route phys_opt may skip paths whose estimated slack is positive.
  # Revisit the actual routed BRAM-to-DSP path without relaxing constraints.
  set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
}
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
  error "impl_1 did not complete: [get_property STATUS [get_runs impl_1]]"
}

open_run impl_1
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1 -nworst 1]]
puts "PLD_INFO: WNS = $wns ns"
report_timing_summary -file "$out_root/timing_summary.rpt"
report_utilization    -file "$out_root/utilization.rpt"
report_drc -file "$out_root/drc.rpt"
if {$pl_core eq "cnn"} {
  set bram36 [llength [get_cells -hier -quiet -filter {REF_NAME == RAMB36E1}]]
  set bram18 [llength [get_cells -hier -quiet -filter {REF_NAME == RAMB18E1}]]
  set dsp [llength [get_cells -hier -quiet -filter {REF_NAME =~ DSP48*}]]
  set luts [llength [get_cells -hier -quiet -filter {REF_NAME =~ LUT*}]]
  if {2*$bram36+$bram18 > 180 || $dsp > 96 || $luts > 32000} {
    error "CNN exceeds B budget: BRAM36=$bram36 BRAM18=$bram18 DSP=$dsp LUT=$luts"
  }
  puts "PLD_INFO: CNN_BUDGET_PASS BRAM36=$bram36 BRAM18=$bram18 DSP=$dsp LUT=$luts"
}

write_hw_platform -fixed -include_bit -force "$out_root/dma_top.xsa"
puts "PLD_INFO: wrote $out_root/dma_top.xsa"
close_project
puts "PLD_INFO: BUILD_OK pl_core=$pl_core"
