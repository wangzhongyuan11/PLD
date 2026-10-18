// One bounded DataMover Full command, 32-bit address / aligned 64-bit payload.
// Reusable for MM2S or S2MM. Completion requires BOTH TLAST and matching status.
// Any protocol/status/timeout fault latches until reset; the owner must also
// quiesce/reset DataMover before reusing memory, not just reset this monitor.
module dma_transfer #(
  parameter integer TIMEOUT_CYCLES=10000000
) (
  input logic aclk,aresetn,
  input logic [31:0] req_address,
  input logic [22:0] req_bytes,
  input logic [3:0] req_tag,
  input logic req_valid,
  output logic req_ready,
  output logic [71:0] cmd_data,
  output logic cmd_valid,
  input logic cmd_ready,
  input logic [7:0] status_data,
  input logic status_valid,
  output logic status_ready,
  input logic [63:0] s_data,
  input logic [7:0] s_keep,
  input logic s_last,s_valid,
  output logic s_ready,
  output logic [63:0] m_data,
  output logic [7:0] m_keep,
  output logic m_last,m_valid,
  input logic m_ready,
  output logic busy,done,fault,
  output logic [7:0] error,
  output logic [31:0] actual_bytes
);
  typedef enum logic [1:0] {IDLE,COMMAND,ACTIVE,FAULT} state_t;
  state_t state;
  logic [31:0] address,expected_bytes;
  logic [3:0] tag;
  logic data_done,status_done;
  logic [31:0] timer;
  logic [7:0] next_error;
  logic [32:0] request_end;
  assign request_end={1'b0,req_address}+{10'd0,req_bytes};
  assign req_ready=aresetn&&state==IDLE;
  // DRR=0 (DRE disabled), EOF=1, DSA=0, INCR=1, BTT.
  assign cmd_data={4'd0,tag,address,1'b0,1'b1,6'd0,1'b1,expected_bytes[22:0]};
  assign cmd_valid=aresetn&&state==COMMAND;
  assign status_ready=aresetn&&state==ACTIVE&&!status_done;
  assign m_data=s_data;assign m_keep=s_keep;assign m_last=s_last;
  assign m_valid=aresetn&&state==ACTIVE&&!data_done&&s_valid;
  assign s_ready=aresetn&&state==ACTIVE&&!data_done&&m_ready;
  assign busy=state!=IDLE;assign fault=state==FAULT;
  always_comb begin
    next_error=error;
    if(state==ACTIVE)begin
      if(s_valid&&s_ready)begin
        if(s_keep!=8'hff||s_last!=(actual_bytes+8==expected_bytes)||actual_bytes+8>expected_bytes)
          next_error|=8'h02;
      end
      if(status_valid&&status_ready)begin
        if(!status_data[7]||status_data[6:4]!=0||status_data[3:0]!=tag)next_error|=8'h04;
      end
      if((data_done&&s_valid)||(status_done&&status_valid))next_error|=8'h10;
    end
    if((state==COMMAND||state==ACTIVE)&&timer>=TIMEOUT_CYCLES-1)next_error|=8'h08;
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)begin state<=IDLE;done<=0;error<=0;actual_bytes<=0;timer<=0;data_done<=0;status_done<=0;end
    else begin
      done<=0;
      case(state)
        IDLE:if(req_valid)begin
          error<=0;actual_bytes<=0;timer<=0;data_done<=0;status_done<=0;
          address<=req_address;expected_bytes<={9'd0,req_bytes};tag<=req_tag;
          if(req_address[2:0]!=0||req_bytes==0||req_bytes[2:0]!=0||request_end[32])begin
            error<=1;state<=FAULT;done<=1;
          end else state<=COMMAND;
        end
        COMMAND:begin
          timer<=timer+1;error<=next_error;
          if(next_error[3])begin state<=FAULT;done<=1;end
          else if(cmd_ready)state<=ACTIVE;
        end
        ACTIVE:begin
          timer<=timer+1;error<=next_error;
          if(s_valid&&s_ready)begin actual_bytes<=actual_bytes+8;if(s_last)data_done<=1;end
          if(status_valid&&status_ready)status_done<=1;
          if(next_error[3])begin state<=FAULT;done<=1;end
          else if(data_done&&status_done)begin state<=next_error==0?IDLE:FAULT;done<=1;end
        end
        FAULT:;
        default:state<=FAULT;
      endcase
    end
  end
  initial if(TIMEOUT_CYCLES<2)$fatal(1,"DMA watchdog must allow at least two cycles");
endmodule
