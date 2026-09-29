// Convert 64-bit command packets into bytes. Low contiguous TKEEP is required;
// only the final beat may be partial. Protocol errors latch until reset.
module axis_byte_unpack (
  input logic aclk, aresetn,
  input logic [63:0] s_axis_tdata,
  input logic [7:0] s_axis_tkeep,
  input logic s_axis_tvalid, s_axis_tlast,
  output logic s_axis_tready,
  output logic [7:0] m_data,
  output logic m_valid, m_last,
  input logic m_ready,
  output logic protocol_error
);
  logic active,last_q;
  logic [63:0] data_q;
  logic [3:0] count,count_q;
  logic [2:0] index;
  always_comb begin
    case(s_axis_tkeep)
      8'h01: count=1;
      8'h03: count=2;
      8'h07: count=3;
      8'h0f: count=4;
      8'h1f: count=5;
      8'h3f: count=6;
      8'h7f: count=7;
      8'hff: count=8;
      default: count=0;
    endcase
  end
  assign s_axis_tready=aresetn && !active && !protocol_error;
  assign m_valid=active && !protocol_error;
  assign m_data=data_q[index*8+:8];
  assign m_last=last_q && ({1'b0,index}==count_q-1'b1);
  always_ff @(posedge aclk) begin
    if (!aresetn) begin active<=0; protocol_error<=0; index<=0; end
    else begin
      if (s_axis_tvalid && s_axis_tready) begin
        if (count==0 || (!s_axis_tlast && count!=8)) protocol_error<=1;
        else begin
          active<=1; data_q<=s_axis_tdata; last_q<=s_axis_tlast; count_q<=count; index<=0;
        end
      end
      if (m_valid && m_ready) begin
        if ({1'b0,index}==count_q-1'b1) active<=0;
        else index<=index+1'b1;
      end
    end
  end
endmodule
