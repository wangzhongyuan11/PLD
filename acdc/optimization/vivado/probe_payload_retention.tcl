# Isolate payload-retention optimization from controller/arithmetic changes.
set root [file normalize [file join [file dirname [info script]] ../../..]]
set out "$root/out/acdc/optimization/executor_payload_probe"
file mkdir $out
set f [open "$root/acdc/optimization/rtl/stream_fork.sv" r]
set source [read $f];close $f
set source [string map [list {logic [WIDTH-1:0] held;} {(* keep="true" *) logic [WIDTH-1:0] held;}] $source]
set f [open "$out/stream_fork.sv" w];puts $f $source;close $f
create_project -in_memory -part xc7z020clg400-1
set files [list "$root/rtl/requant_q24.sv" "$out/stream_fork.sv"]
foreach module {conv_array conv_compute tile_banks tile_conv pool_stream dma_transfer tile_writer parameter_loader input_transform input_loader lut_loader dma_read_arbiter tile_job_queue tile_executor} {
  lappend files "$root/acdc/optimization/rtl/${module}.sv"
}
read_verilog -sv $files
synth_design -mode out_of_context -top tile_executor -generic P=4 -part xc7z020clg400-1
create_clock -period 10.000 [get_ports aclk]
opt_design
report_utilization -hierarchical -file "$out/hierarchy.rpt"
write_checkpoint -force "$out/core.dcp"
write_verilog -force -mode funcsim "$out/netlist.v"
puts "ACDC_PAYLOAD_PROBE_DONE DSP=[llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]"
