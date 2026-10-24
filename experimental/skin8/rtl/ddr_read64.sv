// Single-outstanding 64-bit AXI INCR reader, <=16 beats and no 4KiB crossing.
// Reset only when fabric transactions are quiescent. A system watchdog must
// handle a slave that never responds; this block never abandons an issued read.
module ddr_read64 (
  input logic clk,resetn,
  input logic [31:0] cmd_addr,
  input logic [15:0] cmd_words,
  input logic cmd_valid,
  output logic cmd_ready,done,error,
  output logic [31:0] araddr,
  output logic [7:0] arlen,
  output logic [2:0] arsize,
  output logic [1:0] arburst,
  output logic arvalid,
  input logic arready,
  input logic [63:0] rdata,
  input logic [1:0] rresp,
  input logic rlast,rvalid,
  output logic rready,
  output logic [63:0] m_data,
  output logic m_valid,m_last,
  input logic m_ready
);
  typedef enum logic [2:0] {IDLE,PLAN,ADDRESS,DATA,DRAIN} state_t;
  state_t state;
  logic [15:0] remaining;
  logic [4:0] burst_words,beat;
  logic [9:0] page_words;
  logic [4:0] planned;
  logic [32:0] end_address;
  assign cmd_ready=resetn && state==IDLE;
  assign end_address={1'b0,cmd_addr}+({17'b0,cmd_words}<<3);
  assign page_words=10'd512-{1'b0,araddr[11:3]};
  always_comb begin
    planned=remaining>16 ? 16 : remaining[4:0];
    if(page_words<planned) planned=page_words[4:0];
  end
  assign arsize=3;assign arburst=1;
  assign arvalid=resetn && state==ADDRESS;
  assign m_data=rdata;
  assign m_valid=resetn && state==DATA && rvalid;
  assign m_last=remaining==1;
  assign rready=resetn && (state==DRAIN || (state==DATA && m_ready));
  always_ff @(posedge clk) begin
    if(!resetn) begin state<=IDLE;done<=0;error<=0;end
    else begin
      done<=0;
      case(state)
        IDLE: if(cmd_valid) begin
          error<=0;
          if(cmd_addr[2:0]!=0 || cmd_words==0 || end_address>33'h100000000) begin
            error<=1;done<=1;
          end else begin araddr<=cmd_addr;remaining<=cmd_words;state<=PLAN;end
        end
        PLAN: begin burst_words<=planned;arlen<=planned-1;beat<=0;state<=ADDRESS;end
        ADDRESS: if(arready) state<=DATA;
        DATA: if(rvalid && rready) begin
          remaining<=remaining-1;beat<=beat+1;
          if(rresp!=0) error<=1;
          if(rlast) begin
            if(beat!=burst_words-1 || error || rresp!=0) begin error<=1;done<=1;state<=IDLE;end
            else if(remaining==1) begin done<=1;state<=IDLE;end
            else begin araddr<=araddr+({27'b0,burst_words}<<3);state<=PLAN;end
          end else if(beat==burst_words-1) begin error<=1;state<=DRAIN;end
        end
        DRAIN: if(rvalid && rlast) begin done<=1;state<=IDLE;end
        default: begin state<=IDLE;error<=1;end
      endcase
    end
  end
endmodule
