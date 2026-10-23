`timescale 1ns/1ps
// Icarus/UNISIM initialization regression. Vendor LUT6 uses always @(inputs),
// which can miss declaration initializers propagated before the process waits.
// Drive testbench inputs after time zero; do not force DUT outputs or alter INIT.
module tb_unisim_initialization;
  logic [5:0] declaration_input=0;
  logic [5:0] scheduled_input;
  wire declaration_output,scheduled_output;
  LUT6 #(.INIT(64'hAAAAAAAAAAAAAAAA)) declaration_lut(
    .O(declaration_output),.I0(declaration_input[0]),.I1(declaration_input[1]),
    .I2(declaration_input[2]),.I3(declaration_input[3]),
    .I4(declaration_input[4]),.I5(declaration_input[5]));
  LUT6 #(.INIT(64'hAAAAAAAAAAAAAAAA)) scheduled_lut(
    .O(scheduled_output),.I0(scheduled_input[0]),.I1(scheduled_input[1]),
    .I2(scheduled_input[2]),.I3(scheduled_input[3]),
    .I4(scheduled_input[4]),.I5(scheduled_input[5]));
  initial begin
    #1;scheduled_input=0;
    #1;
    $display("UNISIM_INIT declaration=%b scheduled=%b",declaration_output,scheduled_output);
    if(scheduled_output!==1'b0)$fatal(1,"scheduled input did not initialize LUT");
    scheduled_input=63;declaration_input=63;
    #1;
    if(scheduled_output!==1'b1||declaration_output!==1'b1)$fatal(1,"LUT high transition");
    scheduled_input=0;declaration_input=0;
    #1;
    if(scheduled_output!==1'b0||declaration_output!==1'b0)$fatal(1,"LUT low transition");
    $display("TB_PASS unisim_initialization");$finish;
  end
endmodule
