set root [file normalize [file join [file dirname [info script]] ..]]
set out "$root/out/accel_synth"
file mkdir $out
create_project -in_memory -part xc7z020clg400-1
foreach module {conv_mac_array requant_q24 conv_tile_engine tensor_tile_engine axi_lite_endpoint axis_byte_pack axis_byte_unpack axis_tensor_accel} {
  add_files "$root/rtl/${module}.sv"
}
set fd [open "$out/clock.xdc" w]
puts $fd {create_clock -name aclk -period 10.0 [get_ports aclk]}
close $fd
read_xdc "$out/clock.xdc"
synth_design -top axis_tensor_accel -part xc7z020clg400-1 -mode out_of_context
report_utilization -file "$out/utilization.rpt"
report_timing_summary -file "$out/timing.rpt"
report_timing -max_paths 20 -nworst 1 -file "$out/critical_paths.rpt"
write_checkpoint -force "$out/axis_tensor_accel.dcp"
puts "ACCEL_SYNTH_DONE"
