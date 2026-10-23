set root [file normalize [file join [file dirname [info script]] ../../..]]
set out "$root/out/acdc/optimization/pair_synth"
file mkdir $out
set py "$root/acdc/.venv/Scripts/python.exe"
set guard "$root/acdc/optimization/synth_guard.py"
exec $py $guard begin $out "$root/acdc/optimization/rtl/int8_pair_mul.sv" [info script] $guard
create_project -in_memory -part xc7z020clg400-1
read_verilog -sv "$root/acdc/optimization/rtl/int8_pair_mul.sv"
synth_design -mode out_of_context -top int8_pair_mul -part xc7z020clg400-1
create_clock -period 10.000 [get_ports aclk]
opt_design
report_utilization -file "$out/utilization.rpt"
report_timing_summary -file "$out/timing.rpt"
set dsps [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]
if {$dsps != 1} {error "PAIR_DSP_COUNT expected=1 actual=$dsps"}
write_checkpoint -force "$out/core.dcp"
exec $py $guard finish $out --dsp $dsps
puts "ACDC_PAIR_SYNTH_PASS DSP=$dsps scope=OOC_not_full_system_timing"
