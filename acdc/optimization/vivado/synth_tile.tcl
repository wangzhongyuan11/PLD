set root [file normalize [file join [file dirname [info script]] ../../..]]
set p [lindex $argv 0]
if {$p ni {4 6}} {error "P must be 4 or 6"}
set out "$root/out/acdc/optimization/tile_p${p}_synth"
file mkdir $out
set py "$root/acdc/.venv/Scripts/python.exe"
set guard "$root/acdc/optimization/synth_guard.py"
set files [list "$root/acdc/optimization/rtl/conv_array.sv" "$root/acdc/optimization/rtl/conv_compute.sv" "$root/acdc/optimization/rtl/tile_banks.sv" "$root/acdc/optimization/rtl/tile_conv.sv" "$root/rtl/requant_q24.sv"]
exec $py $guard begin $out {*}$files [info script] $guard
create_project -in_memory -part xc7z020clg400-1
read_verilog -sv $files
synth_design -mode out_of_context -top tile_conv -generic P=$p -part xc7z020clg400-1
create_clock -period 10.000 [get_ports aclk]
opt_design
report_utilization -file "$out/utilization.rpt"
report_timing_summary -file "$out/timing.rpt"
set dsps [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]
set bram36 [llength [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]]
set bram18 [llength [get_cells -hierarchical -filter {REF_NAME == RAMB18E1}]]
set actual [expr {$bram36+0.5*$bram18}]
set expected [expr {$p==4?88:100}]
if {$dsps>220||$actual!=$expected} {error "TILE_RESOURCE_FAIL expected_bram=$expected actual_bram=$actual DSP=$dsps"}
write_checkpoint -force "$out/core.dcp"
exec $py $guard finish $out --dsp $dsps
puts "ACDC_TILE_SYNTH_PASS P=$p DSP=$dsps BRAM36=$actual scope=OOC_not_full_system_timing"
