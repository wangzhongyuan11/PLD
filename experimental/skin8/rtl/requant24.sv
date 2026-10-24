// Full signed INT32 * signed INT32, divide by 2^24, ties-to-even, [-127,127].
// Four 16-bit partial products avoid an unpipelined wide multiply path.
module requant24 #(parameter integer TAG_W=16) (
  input logic clk,resetn,
  input logic signed [31:0] s_acc,s_multiplier,
  input logic [TAG_W-1:0] s_tag,
  input logic s_relu,s_valid,
  output logic s_ready,
  output logic signed [7:0] m_value,
  output logic [TAG_W-1:0] m_tag,
  output logic m_valid,
  input logic m_ready
);
  (* use_dsp="yes" *) logic [31:0] ll;
  (* use_dsp="yes" *) logic signed [32:0] lh,hl;
  (* use_dsp="yes" *) logic signed [31:0] hh;
  (* use_dsp="no" *) logic signed [63:0] product;
  logic v_in,v0,v1,relu_in,relu0,relu1;
  logic signed [31:0] acc_in,mult_in;
  logic [TAG_W-1:0] tag_in,tag0,tag1;
  logic advance;
  logic signed [40:0] rounded;
  assign advance=!m_valid || m_ready;
  assign s_ready=resetn && advance;
  always_comb begin
    rounded=$signed({product[63],product[63:24]});
    if (product[23:0]>24'h800000 || (product[23:0]==24'h800000 && product[24]))
      rounded=rounded+41'sd1;
  end
  always_ff @(posedge clk) begin
    if (!resetn) begin v_in<=0;v0<=0;v1<=0;m_valid<=0;end
    else if (advance) begin
      v_in<=s_valid;v0<=v_in;v1<=v0;m_valid<=v1;
      if (s_valid) begin
        acc_in<=s_acc;mult_in<=s_multiplier;tag_in<=s_tag;relu_in<=s_relu;
      end
      if (v_in) begin
        ll<=acc_in[15:0]*mult_in[15:0];
        lh<=$signed({1'b0,acc_in[15:0]})*$signed(mult_in[31:16]);
        hl<=$signed(acc_in[31:16])*$signed({1'b0,mult_in[15:0]});
        hh<=$signed(acc_in[31:16])*$signed(mult_in[31:16]);
        tag0<=tag_in;relu0<=relu_in;
      end
      if (v0) begin
        product<=$signed({32'b0,ll}) + ($signed({{31{lh[32]}},lh})<<<16) +
                 ($signed({{31{hl[32]}},hl})<<<16) + ($signed({{32{hh[31]}},hh})<<<32);
        tag1<=tag0;relu1<=relu0;
      end
      if (v1) begin
        if (rounded>127) m_value<=8'sd127;
        else if (relu1 && rounded<0) m_value<=0;
        else if (rounded< -127) m_value<= -8'sd127;
        else m_value<=rounded[7:0];
        m_tag<=tag1;
      end
    end
  end
endmodule
