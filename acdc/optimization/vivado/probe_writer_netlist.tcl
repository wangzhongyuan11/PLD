set root [file normalize [file join [file dirname [info script]] ../../..]]
set out "$root/out/acdc/optimization/writer_netlist_probe"
file mkdir $out
open_checkpoint "$root/out/acdc/optimization/writer_p4_synth/core.dcp"
write_verilog -force -mode funcsim "$out/netlist.v"
puts "ACDC_WRITER_NETLIST_EXPORTED"
