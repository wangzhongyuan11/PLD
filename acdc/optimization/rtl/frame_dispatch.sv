// Fetch ordered 64-byte jobs from DDR and retire checked tile completions.
// A frame starts once; PS does not submit individual tiles or layer tensors.
module frame_dispatch #(
  parameter integer TIMEOUT_CYCLES=10000000
) (
  input logic aclk,aresetn,start,
  input logic [31:0] cfg_program,cfg_epoch,
  input logic [15:0] cfg_count,
  output logic busy,done,fault,
  output logic [15:0] error,
  output logic [31:0] error_tag,cycles,
  output logic [15:0] submitted,completed,
  output logic [511:0] descriptor,
  output logic descriptor_valid,
  input logic descriptor_ready,
  input logic [31:0] completion_tag,
  input logic completion_valid,
  output logic completion_ready,
  input logic executor_busy,executor_fault,
  input logic [15:0] executor_error,
  input logic [31:0] executor_error_tag,
  output logic [71:0] cmd_data,
  output logic cmd_valid,
  input logic cmd_ready,
  input logic [7:0] status_data,
  input logic status_valid,
  output logic status_ready,
  input logic [63:0] dma_data,
  input logic [7:0] dma_keep,
  input logic dma_last,dma_valid,
  output logic dma_ready
);
  typedef enum logic [2:0] {IDLE,REQUEST,RECEIVE,CHECK,OFFER,RETIRE,FAULT} state_t;
  state_t state;
  logic [31:0] address,epoch;
  logic [15:0] count;
  logic [2:0] word_index;
  logic [63:0] data;
  logic [7:0] keep,transfer_error;
  logic last,valid,ready,req_ready,transfer_busy,transfer_done,transfer_fault,finished;
  logic [31:0] actual_bytes;
  assign busy=state!=IDLE;
  assign fault=state==FAULT;
  assign descriptor_valid=aresetn&&state==OFFER;
  assign completion_ready=aresetn&&state!=IDLE&&state!=FAULT;
  assign ready=state==RECEIVE;
  dma_transfer #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) fetch(.aclk,.aresetn,
    .req_address(address),.req_bytes(23'd64),.req_tag(4'h4),.req_valid(state==REQUEST),.req_ready,
    .cmd_data,.cmd_valid,.cmd_ready,.status_data,.status_valid,.status_ready,
    .s_data(dma_data),.s_keep(dma_keep),.s_last(dma_last),.s_valid(dma_valid),.s_ready(dma_ready),
    .m_data(data),.m_keep(keep),.m_last(last),.m_valid(valid),.m_ready(ready),
    .busy(transfer_busy),.done(transfer_done),.fault(transfer_fault),.error(transfer_error),.actual_bytes);
  always_ff @(posedge aclk)begin
    if(!aresetn)begin state<=IDLE;done<=0;error<=0;error_tag<=0;cycles<=0;submitted<=0;completed<=0;finished<=0;end
    else begin
      done<=0;
      if(busy&&!fault)cycles<=cycles+1;
      if(transfer_done)finished<=1;
      case(state)
        IDLE:if(start)begin
          cycles<=0;submitted<=0;completed<=0;error<=0;error_tag<=0;finished<=0;
          address<=cfg_program;epoch<=cfg_epoch;count<=cfg_count;word_index<=0;
          if(cfg_count==0||cfg_program[5:0]!=0||{1'b0,cfg_program}+{11'd0,cfg_count,6'd0}>33'h100000000)begin
            state<=FAULT;error<=16'h0701;done<=1;
          end else state<=REQUEST;
        end
        REQUEST:if(req_ready)begin state<=RECEIVE;word_index<=0;end
        RECEIVE:if(valid&&ready)begin
          descriptor[word_index*64+:64]<=data;
          if(word_index==7)state<=CHECK;else word_index<=word_index+1;
        end
        CHECK:if(finished)begin
          finished<=0;
          if(descriptor[415:384]!={16'd0,submitted}||descriptor[511:480]!=epoch)begin
            state<=FAULT;error<=16'h0702;error_tag<=submitted;done<=1;
          end else state<=OFFER;
        end
        OFFER:if(descriptor_ready)begin
          submitted<=submitted+1;
          if(submitted+1==count)state<=RETIRE;
          else begin address<=address+64;state<=REQUEST;end
        end
        RETIRE:if(completed==count&&!executor_busy)begin state<=IDLE;done<=1;end
        FAULT:;
        default:state<=FAULT;
      endcase
      if(completion_valid&&completion_ready)begin
        if(completion_tag!={16'd0,completed}||completed>=submitted)begin
          state<=FAULT;error<=16'h0703;error_tag<=completion_tag;done<=1;
        end else completed<=completed+1;
      end
      if(state!=IDLE&&state!=FAULT)begin
        if(transfer_fault)begin state<=FAULT;error<={8'h07,transfer_error};error_tag<=submitted;done<=1;end
        if(executor_fault)begin state<=FAULT;error<=executor_error;error_tag<=executor_error_tag;done<=1;end
      end
    end
  end
endmodule
