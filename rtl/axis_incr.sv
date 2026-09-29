// Placeholder accelerator IP (stage 2): byte-wise +1 (mod 256) on an
// AXI4-Stream. tvalid/tready/tlast/tkeep handshake is fully honored;
// tkeep passes through unchanged so partial-byte tails (non-multiple-of-8
// DMA lengths) keep their byte enables — S2MM length checks depend on it.
// Output byte i of each beat equals input byte i + 1 (wrapping at 0xFF).
// Deliberately NOT a plain pass-through so the board-side self-test can
// prove data actually traversed PL logic. One register stage of latency.
module axis_incr #(
  parameter int DATA_W = 64  // must be a multiple of 8
) (
  input  logic                  aclk,
  input  logic                  aresetn,

  input  logic [DATA_W-1:0]     s_axis_tdata,
  input  logic [DATA_W/8-1:0]   s_axis_tkeep,
  input  logic                  s_axis_tvalid,
  input  logic                  s_axis_tlast,
  output logic                  s_axis_tready,

  output logic [DATA_W-1:0]     m_axis_tdata,
  output logic [DATA_W/8-1:0]   m_axis_tkeep,
  output logic                  m_axis_tvalid,
  output logic                  m_axis_tlast,
  input  logic                  m_axis_tready
);

  localparam int KEEP_W = DATA_W/8;
  localparam int PAY_W  = DATA_W + KEEP_W + 1;  // tkeep/tlast ride the payload bus

  function automatic logic [DATA_W-1:0] incr_bytes(input logic [DATA_W-1:0] d);
    incr_bytes = '0;
    for (int b = 0; b < KEEP_W; b++)
      incr_bytes[8*b +: 8] = d[8*b +: 8] + 8'd1;
  endfunction

  logic [PAY_W-1:0] s_pay, m_pay;

  assign s_pay = {s_axis_tlast, s_axis_tkeep, incr_bytes(s_axis_tdata)};
  assign {m_axis_tlast, m_axis_tkeep, m_axis_tdata} = m_pay;

  axis_reg_slice #(.DATA_W(PAY_W)) u_slice (
    .aclk        (aclk),
    .aresetn     (aresetn),
    .s_axis_tdata(s_pay),
    .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready),
    .m_axis_tdata(m_pay),
    .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tready(m_axis_tready)
  );

endmodule
