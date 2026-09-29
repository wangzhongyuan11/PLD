# Inference/resource check; full routed BD timing remains a separate gate.
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root out tile_synth]
file mkdir $out
foreach top {conv_tile_engine tensor_tile_engine} {
  create_project -in_memory -part xc7z020clg400-1
  foreach name {conv_mac_array requant_q24 conv_tile_engine tensor_tile_engine} {
    add_files [file join $root rtl ${name}.sv]
  }
  synth_design -top $top -part xc7z020clg400-1 -mode out_of_context
  create_clock -name aclk -period 10.0 [get_ports aclk]
  report_utilization -file [file join $out ${top}_utilization.rpt]
  report_timing_summary -file [file join $out ${top}_timing.rpt]
  set bram36 [llength [get_cells -hier -filter {REF_NAME == RAMB36E1}]]
  set bram18 [llength [get_cells -hier -filter {REF_NAME == RAMB18E1}]]
  set dsp [llength [get_cells -hier -filter {REF_NAME =~ DSP48*}]]
  # These are architecture/inference gates, not substitutes for routed BD timing.
  if {$top eq "conv_tile_engine" && 2*$bram36+$bram18 != 128} {
    error "Feature/weight RAM must infer 64 BRAM36 equivalents; got 36=$bram36 18=$bram18"
  }
  if {$dsp > 64} {error "Unexpected per-engine DSP growth: $top $dsp"}
  write_checkpoint -force [file join $out ${top}.dcp]
  puts "TILE_SYNTH_PASS top=$top dsp=$dsp bram36=$bram36 bram18=$bram18"
  close_project
}
