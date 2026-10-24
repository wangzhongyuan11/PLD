set root [file normalize [file join [file dirname [info script]] ..]]
set top dot128
if {[llength $argv]>0} {set top [lindex $argv 0]}
set out [file join $root out ${top}_ooc_[clock format [clock seconds] -format %Y%m%d_%H%M%S]]
if {[info exists ::env(SKIN8_BUILD_OUT)]} {set out $::env(SKIN8_BUILD_OUT)}
file mkdir $out
cd $out
if {$top eq "stripe_engine" || $top eq "tensor_load" || $top eq "frame_engine"} {
  foreach src [glob [file join $root rtl *.sv]] {read_verilog -sv $src}
} elseif {$top eq "conv_pair_tile" || $top eq "tile_int8"} {
  read_verilog -sv [file join $root rtl dot128.sv]
  read_verilog -sv [file join $root rtl feature_banks.sv]
}
if {$top eq "tile_int8"} {
  read_verilog -sv [file join $root rtl conv_pair_tile.sv]
  read_verilog -sv [file join $root rtl requant24.sv]
  read_verilog -sv [file join $root rtl quant_pair.sv]
}
if {$top ne "stripe_engine" && $top ne "tensor_load" && $top ne "frame_engine"} {read_verilog -sv [file join $root rtl ${top}.sv]}
synth_design -top $top -part xc7z020clg400-1 -mode out_of_context
create_clock -name clk -period 8.000 [get_ports clk]
report_utilization -file utilization_synth.rpt
write_checkpoint -force synth.dcp
opt_design
place_design
phys_opt_design
route_design
report_timing_summary -file timing_routed.rpt
report_utilization -file utilization_routed.rpt
write_checkpoint -force routed.dcp
set fp [open complete.txt w]
puts $fp "OOC only; external interfaces unconstrained; not a deployable bitstream"
close $fp
