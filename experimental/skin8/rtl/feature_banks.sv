// Two ping-pong pages, each split by x parity. A 64-bit word holds 8 channels.
// Address = (y*ceil(width/2) + floor(x/2))*ceil(channels/8) + channel_group.
// Each page/bank is one simple dual-port RAM; no array-wide reset.
// Scheduler owns page locks for the ENTIRE compute interval, not just read beats.
module feature_banks #(parameter integer AW=10) (
  input logic clk, resetn,
  input logic [1:0] page_locked,
  input logic w_page,w_bank,
  input logic [AW-1:0] w_addr,
  input logic [63:0] w_data,
  input logic w_valid,
  output logic w_ready,
  input logic r_page, r_swap,
  input logic [AW-1:0] r_even_addr,r_odd_addr,
  input logic r_valid,
  output logic r_ready,
  output logic [127:0] m_data,
  output logic m_valid,
  input logic m_ready
);
  logic [63:0] q [0:1][0:1];
  logic page_q,swap_q;
  assign w_ready=resetn && !page_locked[w_page];
  assign r_ready=resetn && (!m_valid || m_ready);
  assign m_data=swap_q ? {q[page_q][0],q[page_q][1]} : {q[page_q][1],q[page_q][0]};
  for (genvar p=0;p<2;p++) begin: pages
    for (genvar b=0;b<2;b++) begin: banks
      (* ram_style="block" *) logic [63:0] mem [0:(1<<AW)-1];
      always_ff @(posedge clk) begin
        if (w_valid && w_ready && w_page==p && w_bank==b)
          mem[w_addr]<=w_data;
        if (r_valid && r_ready && r_page==p)
          q[p][b]<=mem[b==0 ? r_even_addr : r_odd_addr];
      end
    end
  end
  always_ff @(posedge clk) begin
    if (!resetn) m_valid<=0;
    else if (r_ready) begin
      m_valid<=r_valid;
      if (r_valid) begin page_q<=r_page;swap_q<=r_swap;end
    end
  end
endmodule
