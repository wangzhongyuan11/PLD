# Compare an explicitly register-backed four-entry metadata/accumulator FIFO
# without editing any source used by a concurrent simulation or build.
set root [file normalize [file join [file dirname [info script]] ../../..]]
set out "$root/out/acdc/optimization/executor_fifo_probe"
file mkdir $out
set f [open "$root/acdc/optimization/rtl/conv_compute.sv" r]
set source [read $f];close $f
set source [string map [list \
  {logic [P*256-1:0] array_acc,acc_fifo[DEPTH];} {logic [P*256-1:0] array_acc; (* ram_style="registers" *) logic [P*256-1:0] acc_fifo[DEPTH];} \
  {logic [META-1:0] array_meta,meta_fifo[DEPTH];} {logic [META-1:0] array_meta; (* ram_style="registers" *) logic [META-1:0] meta_fifo[DEPTH];}] $source]
set f [open "$out/conv_compute.sv" w];puts $f $source;close $f
create_project -in_memory -part xc7z020clg400-1
set files [list "$root/rtl/requant_q24.sv" "$out/conv_compute.sv"]
foreach module {conv_array tile_banks tile_conv stream_fork pool_stream dma_transfer tile_writer parameter_loader input_transform input_loader lut_loader dma_read_arbiter tile_job_queue tile_executor} {
  lappend files "$root/acdc/optimization/rtl/${module}.sv"
}
read_verilog -sv $files
synth_design -mode out_of_context -top tile_executor -generic P=4 -part xc7z020clg400-1
create_clock -period 10.000 [get_ports aclk]
opt_design
report_utilization -hierarchical -file "$out/hierarchy.rpt"
write_checkpoint -force "$out/core.dcp"
write_verilog -force -mode funcsim "$out/netlist.v"
puts "ACDC_FIFO_PROBE_DONE DSP=[llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]"
