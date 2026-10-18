set root [file normalize [file join [file dirname [info script]] ../../..]]
set p [lindex $argv 0]
if {$p ni {4 6}} {error "P must be 4 or 6"}
set py "$root/acdc/.venv/Scripts/python.exe"
set guard "$root/acdc/optimization/synth_guard.py"
foreach top {parameter_loader input_loader} {
  set out "$root/out/acdc/optimization/${top}_p${p}_synth"
  file mkdir $out
  set files [list "$root/acdc/optimization/rtl/dma_transfer.sv" "$root/acdc/optimization/rtl/${top}.sv"]
  if {$top eq "input_loader"} {lappend files "$root/acdc/optimization/rtl/input_transform.sv"}
  exec $py $guard begin $out {*}$files [info script] $guard
  create_project -in_memory -part xc7z020clg400-1
  read_verilog -sv $files
  if {$top eq "input_loader"} {
    synth_design -mode out_of_context -top $top -generic P=$p -part xc7z020clg400-1
  } else {
    synth_design -mode out_of_context -top $top -part xc7z020clg400-1
  }
  create_clock -period 10.000 [get_ports aclk]
  opt_design
  report_utilization -file "$out/utilization.rpt"
  report_timing_summary -file "$out/timing.rpt"
  set dsps [llength [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E1}]]
  if {$dsps!=0} {error "LOADER_UNEXPECTED_DSP top=$top DSP=$dsps"}
  write_checkpoint -force "$out/core.dcp"
  exec $py $guard finish $out --dsp $dsps
  puts "ACDC_LOADER_SYNTH_PASS top=$top P=$p DSP=$dsps scope=OOC_not_full_system_timing"
  close_project
}
