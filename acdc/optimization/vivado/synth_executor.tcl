set root [file normalize [file join [file dirname [info script]] ../../..]]
set p [lindex $argv 0]
if {$p ni {4 6}} {error "P must be 4 or 6"}
set tag [lindex $argv 1]
if {$tag ne "" && ![regexp {^[a-zA-Z0-9_-]+$} $tag]} {error "invalid synthesis tag"}
set suffix [expr {$tag eq "" ? "" : "_$tag"}]
set out "$root/out/acdc/optimization/executor_p${p}_synth$suffix"
file mkdir $out
set py "$root/acdc/.venv/Scripts/python.exe"
set guard "$root/acdc/optimization/synth_guard.py"
set files [list "$root/rtl/requant_q24.sv"]
foreach module {conv_array conv_compute tile_banks tile_conv stream_fork pool_stream dma_transfer tile_writer parameter_loader input_transform input_loader lut_loader dma_read_arbiter tile_job_queue tile_executor} {
  lappend files "$root/acdc/optimization/rtl/${module}.sv"
}
exec $py $guard begin $out {*}$files [info script] $guard
create_project -in_memory -part xc7z020clg400-1
read_verilog -sv $files
synth_design -mode out_of_context -top tile_executor -generic P=$p -part xc7z020clg400-1
create_clock -period 10.000 [get_ports aclk]
opt_design
report_utilization -file "$out/utilization.rpt"
report_timing_summary -file "$out/timing.rpt"
set dsps [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]
set bram36 [llength [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]]
set bram18 [llength [get_cells -hierarchical -filter {REF_NAME == RAMB18E1}]]
set actual [expr {$bram36+0.5*$bram18}]
write_checkpoint -force "$out/core.dcp"
write_verilog -force -mode funcsim "$out/netlist.v"
set expected_dsp [expr {$p==4?148:212}]
set minimum_dsp [expr {$p*32+16}]
if {$dsps<$minimum_dsp||$dsps>$expected_dsp||$actual>125} {error "EXECUTOR_RESOURCE_FAIL BRAM36=$actual DSP=$dsps allowed_dsp=$minimum_dsp..$expected_dsp"}
exec $py $guard finish $out --dsp $dsps
puts "ACDC_EXECUTOR_SYNTH_PASS P=$p DSP=$dsps BRAM36=$actual scope=OOC_not_full_system_timing"
