// Single-stage AXI4-Stream register slice (payload only, 2-FF handshake).
// Throughput: 1 beat/cycle. Adds one cycle of latency.
// Payload fields (e.g. tdata/tlast) are packed by the caller into tdata.
module axis_reg_slice #(
  parameter int DATA_W = 8  // payload width in bits
) (
  input  logic              aclk,
  input  logic              aresetn,

  input  logic [DATA_W-1:0] s_axis_tdata,
  input  logic              s_axis_tvalid,
  output logic              s_axis_tready,

  output logic [DATA_W-1:0] m_axis_tdata,
  output logic              m_axis_tvalid,
  input  logic              m_axis_tready
);

  logic [DATA_W-1:0] data_q;
  logic              valid_q;

  // Accept when empty, or when the held beat leaves this cycle.
  assign s_axis_tready = !valid_q || m_axis_tready;
  assign m_axis_tvalid = valid_q;
  assign m_axis_tdata  = data_q;

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      valid_q <= 1'b0;
    end else if (s_axis_tvalid && s_axis_tready) begin
      data_q  <= s_axis_tdata;
      valid_q <= 1'b1;
    end else if (m_axis_tvalid && m_axis_tready) begin
      valid_q <= 1'b0;
    end
  end

endmodule
