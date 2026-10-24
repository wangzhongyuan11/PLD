// Elastic 128->64 serializer. An odd final pixel consumes one output word.
module pair_to_words (
  input logic clk,resetn,
  input logic [127:0] s_data,
  input logic s_second,s_last,s_valid,
  output logic s_ready,
  output logic [63:0] m_data,
  output logic m_last,m_valid,
  input logic m_ready
);
  logic [127:0] data_q;
  logic second_q,last_q,upper,full;
  logic final_word;
  assign final_word=upper || !second_q;
  assign s_ready=resetn && (!full || (m_ready && final_word));
  assign m_data=upper ? data_q[127:64] : data_q[63:0];
  assign m_last=last_q && final_word;
  assign m_valid=resetn && full;
  always_ff @(posedge clk) begin
    if(!resetn) begin full<=0;upper<=0;end
    else begin
      if(m_valid && m_ready) begin
        if(final_word) begin full<=0;upper<=0;end
        else upper<=1;
      end
      if(s_valid && s_ready) begin data_q<=s_data;second_q<=s_second;last_q<=s_last;full<=1;upper<=0;end
    end
  end
endmodule
