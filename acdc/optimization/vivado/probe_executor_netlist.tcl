# Diagnostic: retain hierarchy and export the actual functional netlist so
# an unexpectedly trimmed arithmetic path cannot be accepted by resource count.
set root [file normalize [file join [file dirname [info script]] ../../..]]
set out "$root/out/acdc/optimization/executor_netlist_probe"
file mkdir $out
create_project -in_memory -part xc7z020clg400-1
set files [list "$root/rtl/requant_q24.sv"]
foreach module {conv_array conv_compute tile_banks tile_conv stream_fork pool_stream dma_transfer tile_writer parameter_loader input_transform input_loader lut_loader dma_read_arbiter tile_job_queue tile_executor} {
  lappend files "$root/acdc/optimization/rtl/${module}.sv"
}
read_verilog -sv $files
synth_design -mode out_of_context -top tile_executor -generic P=4 -flatten_hierarchy none -part xc7z020clg400-1
create_clock -period 10.000 [get_ports aclk]
opt_design
report_utilization -hierarchical -file "$out/hierarchy.rpt"
write_checkpoint -force "$out/core.dcp"
write_verilog -force -mode funcsim "$out/netlist.v"
puts "ACDC_NETLIST_PROBE_DONE DSP=[llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]"
