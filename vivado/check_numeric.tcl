# Resource/inference smoke check for arithmetic primitives, not routed timing.
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root out numeric_synth]
file mkdir $out
foreach top {requant_q24 conv_mac_array} {
  create_project -in_memory -part xc7z020clg400-1
  add_files [file join $root rtl ${top}.sv]
  synth_design -top $top -part xc7z020clg400-1 -mode out_of_context
  create_clock -name aclk -period 10.0 [get_ports aclk]
  report_utilization -file [file join $out ${top}_utilization.rpt]
  report_timing_summary -file [file join $out ${top}_timing.rpt]
  write_checkpoint -force [file join $out ${top}.dcp]
  puts "NUMERIC_SYNTH_PASS top=$top dsp=[llength [get_cells -hier -filter {REF_NAME =~ DSP48*}]]"
  close_project
}
