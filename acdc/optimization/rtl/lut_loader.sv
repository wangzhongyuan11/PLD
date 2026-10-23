// Tiny model-dependent 512-byte CAT table; byte-serial programming is deliberate
// here (512 clocks per table), separate from the wide activation/weight paths.
module lut_loader #(
  parameter integer TIMEOUT_CYCLES=10000000
) (
  input logic aclk,aresetn,start,
  input logic [31:0] cfg_address,
  output logic busy,done,
  output logic [7:0] error,
  output logic [71:0] cmd_data,
  output logic cmd_valid,
  input logic cmd_ready,
  input logic [7:0] status_data,
  input logic status_valid,
  output logic status_ready,
  input logic [63:0] dma_data,
  input logic [7:0] dma_keep,
  input logic dma_last,dma_valid,
  output logic dma_ready,
  output logic lut_we,
  output logic [8:0] lut_addr,
  output logic [7:0] lut_data
);
  typedef enum logic [2:0] {IDLE,REQUEST,RECEIVE,SEND,FINISH,FAULT} state_t;
  state_t state;
  logic [31:0] address;
  logic [63:0] held,data;
  logic [7:0] keep;
  logic [5:0] word_index;
  logic [2:0] byte_index;
  logic data_last,data_valid,data_ready,request_ready,transfer_busy,transfer_done,transfer_fault,finished;
  logic [7:0] transfer_error;
  logic [31:0] actual_bytes;
  assign busy=state!=IDLE;
  assign data_ready=state==RECEIVE;
  assign lut_we=state==SEND;assign lut_addr={word_index,byte_index};
  assign lut_data=held[byte_index*8+:8];
  dma_transfer #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) transfer(.aclk,.aresetn,.req_address(address),
    .req_bytes(23'd512),.req_tag(4'h3),.req_valid(state==REQUEST),.req_ready(request_ready),
    .cmd_data,.cmd_valid,.cmd_ready,.status_data,.status_valid,.status_ready,
    .s_data(dma_data),.s_keep(dma_keep),.s_last(dma_last),.s_valid(dma_valid),.s_ready(dma_ready),
    .m_data(data),.m_keep(keep),.m_last(data_last),.m_valid(data_valid),.m_ready(data_ready),
    .busy(transfer_busy),.done(transfer_done),.fault(transfer_fault),.error(transfer_error),.actual_bytes);
  always_ff @(posedge aclk)begin
    if(!aresetn)begin state<=IDLE;done<=0;error<=0;finished<=0;end
    else begin
      done<=0;
      if(transfer_done)finished<=1;
      case(state)
        IDLE:if(start)begin
          error<=0;finished<=0;address<=cfg_address;word_index<=0;byte_index<=0;
          if(cfg_address[2:0]!=0)begin state<=FAULT;error<=1;done<=1;end
          else state<=REQUEST;
        end
        REQUEST:if(request_ready)state<=RECEIVE;
        RECEIVE:if(data_valid&&data_ready)begin held<=data;byte_index<=0;state<=SEND;end
        SEND:begin
          if(byte_index==7)begin
            if(word_index==63)state<=FINISH;
            else begin word_index<=word_index+1;state<=RECEIVE;end
          end else byte_index<=byte_index+1;
        end
        FINISH:if(finished)begin state<=IDLE;done<=1;end
        FAULT:;
        default:state<=FAULT;
      endcase
      if(state!=IDLE&&state!=FAULT&&transfer_fault)begin state<=FAULT;error<=8'h80|transfer_error;done<=1;end
    end
  end
endmodule
