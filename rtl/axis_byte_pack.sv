// Convert a byte stream into 64-bit AXIS packets. Output TLAST comes from the
// last RESULT byte, independent of input command packet lengths.
module axis_byte_pack (
  input logic aclk, aresetn,
  input logic [7:0] s_data,
  input logic s_valid, s_last,
  output logic s_ready,
  output logic [63:0] m_axis_tdata,
  output logic [7:0] m_axis_tkeep,
  output logic m_axis_tvalid, m_axis_tlast,
  input logic m_axis_tready
);
  logic [63:0] partial,next_word;
  logic [2:0] index;
  assign s_ready=aresetn && (!m_axis_tvalid || m_axis_tready);
  always_comb begin
    next_word=partial;
    next_word[index*8+:8]=s_data;
  end
  always_ff @(posedge aclk) begin
    if (!aresetn) begin m_axis_tvalid<=0; index<=0; partial<=0; end
    else begin
      if (m_axis_tvalid && m_axis_tready) m_axis_tvalid<=0;
      if (s_valid && s_ready) begin
        if (index==7 || s_last) begin
          m_axis_tdata<=next_word;
          m_axis_tkeep<=8'hff>>(7-index);
          m_axis_tlast<=s_last;
          m_axis_tvalid<=1;
          index<=0; partial<=0;
        end else begin index<=index+1'b1; partial<=next_word; end
      end
    end
  end
endmodule
