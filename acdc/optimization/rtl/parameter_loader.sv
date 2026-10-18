// Read one Co8 group: 64-byte bias/M header followed by tap_count*64 weights.
// Weight word order: Ci8,Ky,Kx,Co_lane, then eight signed input-channel bytes.
module parameter_loader #(
  parameter integer TIMEOUT_CYCLES=10000000
) (
  input logic aclk,aresetn,start,
  input logic [31:0] cfg_address,
  input logic [7:0] cfg_ci,
  input logic cfg_kernel3,cfg_slot,
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
  output logic weight_slot,
  output logic [2:0] weight_bank,
  output logic [7:0] weight_addr,
  output logic [63:0] weight_data,
  output logic weight_valid,
  input logic weight_ready,
  output logic param_slot,param_valid,
  output logic [255:0] param_bias,param_multiplier,
  input logic param_ready
);
  typedef enum logic [2:0] {IDLE,PREPARE,REQUEST,HEADER,COMMIT,WEIGHTS,FINISH,FAULT} state_t;
  state_t state;
  logic [31:0] address;
  logic [7:0] ci;
  logic kernel3,slot;
  logic [8:0] taps;
  logic [10:0] weight_index;
  logic [2:0] header_index;
  logic [511:0] header;
  logic [63:0] data;
  logic [7:0] keep;
  logic data_last,data_valid,data_ready,request_ready,transfer_busy,transfer_done,transfer_fault,finished;
  logic [7:0] transfer_error;
  logic [31:0] actual_bytes;
  assign busy=state!=IDLE;
  assign weight_slot=slot;assign param_slot=slot;
  assign weight_bank=weight_index[2:0];assign weight_addr=weight_index[10:3];
  assign weight_data=data;assign weight_valid=state==WEIGHTS&&data_valid;
  assign param_bias=header[255:0];assign param_multiplier=header[511:256];
  assign param_valid=state==COMMIT;
  assign data_ready=state==HEADER||(state==WEIGHTS&&weight_ready);
  dma_transfer #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) transfer(.aclk,.aresetn,
    .req_address(address),.req_bytes({8'd0,taps,6'd0}+23'd64),.req_tag(4'h1),
    .req_valid(state==REQUEST),.req_ready(request_ready),
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
          error<=0;finished<=0;address<=cfg_address;ci<=cfg_ci;kernel3<=cfg_kernel3;slot<=cfg_slot;
          if(cfg_ci==0||cfg_ci>192||cfg_address[2:0]!=0)begin state<=FAULT;error<=1;done<=1;end
          else state<=PREPARE;
        end
        PREPARE:begin taps<=(( {1'b0,ci}+7)>>3)*(kernel3?9:1);state<=REQUEST;end
        REQUEST:if(request_ready)begin header_index<=0;state<=HEADER;end
        HEADER:if(data_valid&&data_ready)begin
          header[header_index*64+:64]<=data;
          if(header_index==7)state<=COMMIT;else header_index<=header_index+1;
        end
        COMMIT:if(param_ready)begin weight_index<=0;state<=WEIGHTS;end
        WEIGHTS:if(data_valid&&data_ready)begin
          if(weight_index+1==taps*8)state<=FINISH;
          else weight_index<=weight_index+1;
        end
        FINISH:if(finished)begin state<=IDLE;done<=1;end
        FAULT:;
        default:state<=FAULT;
      endcase
      if(state!=IDLE&&state!=FAULT&&transfer_fault)begin state<=FAULT;error<=8'h80|transfer_error;done<=1;end
    end
  end
endmodule
