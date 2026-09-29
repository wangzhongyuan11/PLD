# Toolchain smoke test (stage 0b): pure-RTL project through synth -> impl ->
# bitstream on an installed Artix-7 WebPACK part. Proves the batch-mode
# build path and license work; the Zynq-7000 device pack is missing from
# this Vivado install (see AGENTS.md), so the real BD target is blocked.
#   vivado -mode batch -source smoke_synth.tcl -tclargs <out_root>
set out_root [file normalize [lindex $argv 0]]
set repo_root [file normalize [file join [file dirname [info script]] ..]]

create_project -force smoke "$out_root/proj" -part xc7a35tcpg236-1
add_files "$repo_root/rtl/axis_incr.sv"
add_files "$repo_root/rtl/axis_reg_slice.sv"
set_property top axis_incr [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
  error "smoke impl failed: [get_property STATUS [get_runs impl_1]]"
}
open_run impl_1
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1 -nworst 1]]
puts "PLD_INFO: smoke WNS = $wns ns"
close_project
puts "PLD_INFO: SMOKE_BIT_OK"
