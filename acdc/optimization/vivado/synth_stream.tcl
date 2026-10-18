set root [file normalize [file join [file dirname [info script]] ../../..]]
set py "$root/acdc/.venv/Scripts/python.exe"
set guard "$root/acdc/optimization/synth_guard.py"
foreach top {pool_stream input_transform} {
  set out "$root/out/acdc/optimization/${top}_synth"
  file mkdir $out
  set source "$root/acdc/optimization/rtl/${top}.sv"
  exec $py $guard begin $out $source [info script] $guard
  create_project -in_memory -part xc7z020clg400-1
  read_verilog -sv $source
  synth_design -mode out_of_context -top $top -part xc7z020clg400-1
  create_clock -period 10.000 [get_ports aclk]
  opt_design
  report_utilization -file "$out/utilization.rpt"
  report_timing_summary -file "$out/timing.rpt"
  set dsps [llength [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E1}]]
  if {$dsps!=0} {error "STREAM_UNEXPECTED_DSP top=$top DSP=$dsps"}
  write_checkpoint -force "$out/core.dcp"
  exec $py $guard finish $out --dsp $dsps
  puts "ACDC_STREAM_SYNTH_PASS top=$top DSP=$dsps scope=OOC_not_full_system_timing"
  close_project
}
