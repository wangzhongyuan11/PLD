set root [file normalize [file join [file dirname [info script]] ../..]]
set out "$root/out/acdc/mac128_synth"
file mkdir $out
create_project -in_memory -part xc7z020clg400-1
read_verilog -sv "$root/acdc/performance_ip/conv_mac_128.sv"
synth_design -mode out_of_context -top conv_mac_128 -part xc7z020clg400-1
create_clock -period 8.000 [get_ports aclk]
opt_design
report_utilization -file "$out/utilization.rpt"
report_timing_summary -file "$out/timing.rpt"
write_checkpoint -force "$out/mac128.dcp"
puts "ACDC_MAC128_SYNTH_PASS"
