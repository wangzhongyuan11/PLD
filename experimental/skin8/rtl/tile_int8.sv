// Complete INT8 convolution tile interface. Descriptor accepted only while idle.
// Halo packing and per-layer input scale mapping are responsibilities of feeder.
module tile_int8 #(parameter integer AW=12) (
  input logic clk,resetn,
  input logic start,cfg_page,cfg_k3,cfg_relu,
  input logic [8:0] cfg_iw,cfg_ih,
  input logic [4:0] cfg_ci_groups,
  input logic [255:0] cfg_bias,cfg_multiplier,
  output logic busy,error,done,
  input logic f_page,f_bank,
  input logic [AW-1:0] f_addr,
  input logic [63:0] f_data,
  input logic f_valid,
  output logic f_ready,
  input logic [2:0] w_lane,
  input logic [7:0] w_addr,
  input logic [63:0] w_data,
  input logic w_valid,
  output logic w_ready,
  output logic [127:0] m_data,
  output logic [15:0] m_tag,
  output logic m_last,m_second_valid,m_overflow,m_valid,
  input logic m_ready
);
  logic conv_busy,conv_error,conv_done;
  logic [511:0] acc;
  logic [15:0] tag;
  logic last,second_valid,overflow,valid,ready;
  logic [255:0] multiplier;
  logic relu;
  conv_pair_tile #(.AW(AW)) conv (
    .clk,.resetn,.start(start && !busy),.cfg_page,.cfg_k3,.cfg_iw,.cfg_ih,.cfg_ci_groups,.cfg_bias,
    .busy(conv_busy),.error(conv_error),.done(conv_done),
    .f_page,.f_bank,.f_addr,.f_data,.f_valid,.f_ready,.w_lane,.w_addr,.w_data,.w_valid,.w_ready,
    .m_acc(acc),.m_tag(tag),.m_last(last),.m_second_valid(second_valid),.m_overflow(overflow),.m_valid(valid),.m_ready(ready)
  );
  quant_pair quant (
    .clk,.resetn,.s_acc(acc),.s_multiplier(multiplier),.s_tag(tag),.s_relu(relu),
    .s_last(last),.s_second_valid(second_valid),.s_overflow(overflow),.s_valid(valid),.s_ready(ready),
    .m_data,.m_tag,.m_last,.m_second_valid,.m_overflow,.m_valid,.m_ready
  );
  assign error=conv_error;
  always_ff @(posedge clk) begin
    if(!resetn) begin busy<=0;done<=0;end
    else begin
      done<=0;
      if(start && !busy) begin busy<=1;multiplier<=cfg_multiplier;relu<=cfg_relu;end
      else if(conv_error) busy<=0;
      if(m_valid && m_ready && m_last) begin busy<=0;done<=1;end
    end
  end
endmodule
