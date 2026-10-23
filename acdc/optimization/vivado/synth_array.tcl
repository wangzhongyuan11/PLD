set root [file normalize [file join [file dirname [info script]] ../../..]]
set p [lindex $argv 0]
if {$p ni {4 6}} {error "P must be 4 or 6"}
set out "$root/out/acdc/optimization/array_p${p}_synth"
file mkdir $out
set py "$root/acdc/.venv/Scripts/python.exe"
set guard "$root/acdc/optimization/synth_guard.py"
exec $py $guard begin $out "$root/acdc/optimization/rtl/conv_array.sv" [info script] $guard
create_project -in_memory -part xc7z020clg400-1
read_verilog -sv "$root/acdc/optimization/rtl/conv_array.sv"
synth_design -mode out_of_context -top conv_array -generic P=$p -part xc7z020clg400-1
create_clock -period 10.000 [get_ports aclk]
opt_design
report_utilization -file "$out/utilization.rpt"
report_timing_summary -file "$out/timing.rpt"
set dsps [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]
if {$dsps != $p*32} {error "ARRAY_DSP_COUNT expected=[expr {$p*32}] actual=$dsps"}
write_checkpoint -force "$out/core.dcp"
exec $py $guard finish $out --dsp $dsps
puts "ACDC_ARRAY_SYNTH_PASS P=$p DSP=$dsps scope=OOC_not_full_system_timing"
