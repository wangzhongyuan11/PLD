// Reuse two exact quantizers over an 8-channel/two-pixel accumulator group.
// Eight DSPs instead of 64; upstream backpressure covers short head dot products.
module quant_pair (
  input logic clk,resetn,
  input logic [511:0] s_acc,
  input logic [255:0] s_multiplier,
  input logic [15:0] s_tag,
  input logic s_relu,s_last,s_second_valid,s_overflow,s_valid,
  output logic s_ready,
  output logic [127:0] m_data,
  output logic [15:0] m_tag,
  output logic m_last,m_second_valid,m_overflow,m_valid,
  input logic m_ready
);
  logic busy,feeding;
  logic [2:0] index;
  logic [511:0] acc_q;
  logic [255:0] mult_q;
  logic [15:0] tag_q;
  logic relu_q,last_q,second_q,overflow_q;
  logic [1:0] q_in_ready,q_valid;
  logic signed [7:0] q_value [0:1];
  logic [2:0] q_tag [0:1];
  logic q_ready;
  assign s_ready=resetn && !busy && (!m_valid || m_ready);
  assign q_ready=!m_valid || m_ready;
  for(genvar p=0;p<2;p++) begin: quantizers
    requant24 #(.TAG_W(3)) quant (
      .clk,.resetn,.s_acc(acc_q[(p*8+index)*32+:32]),.s_multiplier(mult_q[index*32+:32]),
      .s_tag(index),.s_relu(relu_q),.s_valid(feeding),.s_ready(q_in_ready[p]),
      .m_value(q_value[p]),.m_tag(q_tag[p]),.m_valid(q_valid[p]),.m_ready(q_ready)
    );
  end
  always_ff @(posedge clk) begin
    if(!resetn) begin busy<=0;feeding<=0;m_valid<=0;end
    else begin
      if(m_valid && m_ready) m_valid<=0;
      if(s_valid && s_ready) begin
        acc_q<=s_acc;mult_q<=s_multiplier;tag_q<=s_tag;relu_q<=s_relu;
        last_q<=s_last;second_q<=s_second_valid;overflow_q<=s_overflow;
        busy<=1;feeding<=1;index<=0;
      end
      if(feeding && (&q_in_ready)) begin
        index<=index+1;
        if(index==7) feeding<=0;
      end
      if((&q_valid) && q_ready) begin
        m_data[q_tag[0]*8+:8]<=q_value[0];
        m_data[(q_tag[1]+8)*8+:8]<=q_value[1];
        if(q_tag[0]==7) begin
          m_valid<=1;busy<=0;m_tag<=tag_q;
          m_last<=last_q;m_second_valid<=second_q;m_overflow<=overflow_q;
        end
      end
    end
  end
endmodule
