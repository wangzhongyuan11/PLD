set root [file normalize [file join [file dirname [info script]] ../..]]
set out "$root/out/acdc/core128_synth"
file mkdir $out
create_project -in_memory -part xc7z020clg400-1
foreach module {requant_q24 axi_lite_endpoint axis_byte_pack axis_byte_unpack} {
  read_verilog -sv "$root/rtl/${module}.sv"
}
read_verilog -sv "$root/acdc/rtl/tensor_tile_engine_acdc.sv"
foreach module {conv_mac_128 conv_tile_engine_acdc_128 axis_tensor_accel_acdc_fast} {
  read_verilog -sv "$root/acdc/performance_ip/${module}.sv"
}
synth_design -mode out_of_context -top axis_tensor_accel_acdc_fast -part xc7z020clg400-1
create_clock -period 8.000 [get_ports aclk]
report_utilization -file "$out/utilization.rpt"
report_utilization -hierarchical -file "$out/hierarchy.rpt"
set r36 [llength [get_cells -hier -filter {REF_NAME == RAMB36E1}]]
set r18 [llength [get_cells -hier -filter {REF_NAME == RAMB18E1}]]
set dsp [llength [get_cells -hier -filter {REF_NAME =~ DSP48*}]]
if {2*$r36+$r18>234 || $dsp>180} {error "core alone exceeds reserved device budget: RAMB36=$r36 RAMB18=$r18 DSP=$dsp"}
write_checkpoint -force "$out/core128.dcp"
puts "ACDC_CORE128_CAPACITY_PASS RAMB36=$r36 RAMB18=$r18 DSP=$dsp"
