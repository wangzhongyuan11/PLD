# Compare a preserved writer boundary with the unchanged standalone writer DCP.
set root [file normalize [file join [file dirname [info script]] ../../..]]
set out "$root/out/acdc/optimization/executor_boundary_probe"
file mkdir $out
set f [open "$root/acdc/optimization/rtl/tile_executor.sv" r]
set source [read $f];close $f
set source [string map [list {tile_writer #(} {(* dont_touch="yes" *) tile_writer #(}] $source]
set f [open "$out/tile_executor.sv" w];puts $f $source;close $f
create_project -in_memory -part xc7z020clg400-1
set files [list "$root/rtl/requant_q24.sv" "$out/tile_executor.sv"]
foreach module {conv_array conv_compute tile_banks tile_conv stream_fork pool_stream dma_transfer tile_writer parameter_loader input_transform input_loader lut_loader dma_read_arbiter tile_job_queue} {
  lappend files "$root/acdc/optimization/rtl/${module}.sv"
}
read_verilog -sv $files
synth_design -mode out_of_context -top tile_executor -generic P=4 -part xc7z020clg400-1
create_clock -period 10.000 [get_ports aclk]
opt_design
report_utilization -hierarchical -file "$out/hierarchy.rpt"
write_checkpoint -force "$out/core.dcp"
write_verilog -force -mode funcsim "$out/netlist.v"
puts "ACDC_BOUNDARY_PROBE_DONE DSP=[llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]"
