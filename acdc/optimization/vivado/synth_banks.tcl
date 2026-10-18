set root [file normalize [file join [file dirname [info script]] ../../..]]
set p [lindex $argv 0]
if {$p ni {4 6}} {error "P must be 4 or 6"}
set words [expr {$p==4?4608:3584}]
set out "$root/out/acdc/optimization/banks_p${p}_synth"
file mkdir $out
set py "$root/acdc/.venv/Scripts/python.exe"
set guard "$root/acdc/optimization/synth_guard.py"
set source "$root/acdc/optimization/rtl/tile_banks.sv"
exec $py $guard begin $out $source [info script] $guard
create_project -in_memory -part xc7z020clg400-1
read_verilog -sv $source
synth_design -mode out_of_context -top tile_banks -generic [list P=$p WORDS=$words] -part xc7z020clg400-1
create_clock -period 10.000 [get_ports aclk]
opt_design
report_utilization -file "$out/utilization.rpt"
report_timing_summary -file "$out/timing.rpt"
set dsps [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]
set bram36 [llength [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]]
set bram18 [llength [get_cells -hierarchical -filter {REF_NAME == RAMB18E1}]]
set actual [expr {$bram36+0.5*$bram18}]
set expected [expr {$p==4?72:84}]
if {$actual != $expected || $dsps != 0} {error "BANK_MAPPING expected=$expected actual=$actual DSP=$dsps"}
write_checkpoint -force "$out/core.dcp"
exec $py $guard finish $out --dsp $dsps
puts "ACDC_BANKS_SYNTH_PASS P=$p BRAM36_equivalent=$actual scope=OOC_not_full_system_timing"
