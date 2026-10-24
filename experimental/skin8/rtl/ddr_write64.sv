// Single-outstanding AXI INCR writer. A command covers aligned 64-bit words.
// Data/strobes remain owned by the producer until s_ready. Error stops further
// bursts after accepting BRESP; the frame controller must discard remaining data.
module ddr_write64 (
  input logic clk,resetn,
  input logic [31:0] cmd_addr,
  input logic [15:0] cmd_words,
  input logic cmd_valid,
  output logic cmd_ready,done,error,
  input logic [63:0] s_data,
  input logic [7:0] s_strb,
  input logic s_valid,
  output logic s_ready,
  output logic [31:0] awaddr,
  output logic [7:0] awlen,
  output logic [2:0] awsize,
  output logic [1:0] awburst,
  output logic awvalid,
  input logic awready,
  output logic [63:0] wdata,
  output logic [7:0] wstrb,
  output logic wlast,wvalid,
  input logic wready,
  input logic [1:0] bresp,
  input logic bvalid,
  output logic bready
);
  typedef enum logic [2:0] {IDLE,PLAN,ADDRESS,DATA,RESPONSE} state_t;
  state_t state;
  logic [15:0] remaining;
  logic [4:0] burst_words,beat,planned;
  logic [9:0] page_words;
  logic [32:0] end_address;
  assign cmd_ready=resetn && state==IDLE;
  assign end_address={1'b0,cmd_addr}+({17'b0,cmd_words}<<3);
  assign page_words=10'd512-{1'b0,awaddr[11:3]};
  always_comb begin
    planned=remaining>16 ? 16 : remaining[4:0];
    if(page_words<planned) planned=page_words[4:0];
  end
  assign awsize=3;assign awburst=1;
  assign awvalid=resetn && state==ADDRESS;
  assign wdata=s_data;assign wstrb=s_strb;
  assign wvalid=resetn && state==DATA && s_valid;
  assign s_ready=resetn && state==DATA && wready;
  assign wlast=beat==burst_words-1;
  assign bready=resetn && state==RESPONSE;
  always_ff @(posedge clk) begin
    if(!resetn) begin state<=IDLE;done<=0;error<=0;end
    else begin
      done<=0;
      case(state)
        IDLE: if(cmd_valid) begin
          error<=0;
          if(cmd_addr[2:0]!=0 || cmd_words==0 || end_address>33'h100000000) begin error<=1;done<=1;end
          else begin awaddr<=cmd_addr;remaining<=cmd_words;state<=PLAN;end
        end
        PLAN: begin burst_words<=planned;awlen<=planned-1;beat<=0;state<=ADDRESS;end
        ADDRESS: if(awready) state<=DATA;
        DATA: if(wvalid && wready) begin
          remaining<=remaining-1;beat<=beat+1;
          if(wlast) state<=RESPONSE;
        end
        RESPONSE: if(bvalid) begin
          if(bresp!=0) begin error<=1;done<=1;state<=IDLE;end
          else if(remaining==0) begin done<=1;state<=IDLE;end
          else begin awaddr<=awaddr+({27'b0,burst_words}<<3);state<=PLAN;end
        end
        default: begin state<=IDLE;error<=1;end
      endcase
    end
  end
endmodule
