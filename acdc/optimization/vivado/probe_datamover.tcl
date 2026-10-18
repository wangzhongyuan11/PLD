# Resolve the installed IP configuration before the full network BD exists.
# This generates a local IP instance only; it neither builds nor loads an overlay.
set root [file normalize [file join [file dirname [info script]] ../../..]]
set out "$root/out/acdc/optimization/datamover_probe"
file mkdir $out
create_project -force datamover_probe "$out/project" -part xc7z020clg400-1
create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 -module_name acdc_datamover
set ip [get_ips acdc_datamover]
set_property -dict [list CONFIG.c_include_mm2s {Full} CONFIG.c_include_s2mm {Full}] $ip
set_property -dict [list \
  CONFIG.c_m_axi_mm2s_data_width {64} CONFIG.c_m_axis_mm2s_tdata_width {64} \
  CONFIG.c_m_axi_s2mm_data_width {64} CONFIG.c_s_axis_s2mm_tdata_width {64} \
  CONFIG.c_addr_width {32} \
  CONFIG.c_include_mm2s_dre {0} CONFIG.c_include_s2mm_dre {0} \
  CONFIG.c_mm2s_burst_size {16} CONFIG.c_s2mm_burst_size {16} \
  CONFIG.c_mm2s_btt_used {23} CONFIG.c_s2mm_btt_used {23} \
] $ip
generate_target instantiation_template $ip
set f [open "$out/configuration.txt" w]
puts $f "Vivado=[version -short] part=xc7z020clg400-1 IP=[get_property IPDEF $ip]"
foreach property [lsort [list_property $ip]] {
  if {[string match CONFIG.* $property]} {puts $f "$property=[get_property $property $ip]"}
}
close $f
puts "ACDC_DATAMOVER_PROBE_PASS output=$out scope=IP_configuration_only"
