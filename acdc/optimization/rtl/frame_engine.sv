// DDR program -> queued input/CAT/parameters -> Conv/Pool -> DDR, one frame.
// No intermediate tensors or per-layer commands are injected by the PS.
module frame_engine #(
  parameter integer P=4,
  parameter integer TIMEOUT_CYCLES=10000000
) (
  input logic aclk,aresetn,start,clear_cache,
  input logic [31:0] cfg_program,cfg_epoch,
  input logic [15:0] cfg_count,
  output logic busy,done,fault,
  output logic [15:0] error,submitted,completed,
  output logic [31:0] error_tag,cycles,input_hits,parameter_hits,overlap_cycles,
  output logic [71:0] read_cmd_data,
  output logic read_cmd_valid,
  input logic read_cmd_ready,
  input logic [63:0] read_data,
  input logic [7:0] read_keep,read_status_data,
  input logic read_last,read_valid,read_status_valid,
  output logic read_ready,read_status_ready,
  output logic [71:0] write_cmd_data,
  output logic write_cmd_valid,
  input logic write_cmd_ready,
  output logic [63:0] write_data,
  output logic [7:0] write_keep,
  output logic write_last,write_valid,
  input logic write_ready,
  input logic [7:0] write_status_data,
  input logic write_status_valid,
  output logic write_status_ready
);
  logic [511:0] descriptor;logic descriptor_valid,descriptor_ready;
  logic [31:0] completion_tag,executor_error_tag;
  logic completion_valid,completion_ready,executor_busy,executor_fault;
  logic [15:0] executor_error;
  logic [143:0] c_cmd;logic [1:0] c_cmd_valid,c_cmd_ready,c_last,c_valid,c_ready,c_status_valid,c_status_ready;
  logic [127:0] c_data;logic [15:0] c_keep,c_status;
  logic arbiter_busy,arbiter_error;
  frame_dispatch #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) dispatch(.aclk,.aresetn,.start,.cfg_program,.cfg_epoch,.cfg_count,
    .busy,.done,.fault,.error,.error_tag,.cycles,.submitted,.completed,
    .descriptor,.descriptor_valid,.descriptor_ready,.completion_tag,.completion_valid,.completion_ready,
    .executor_busy,.executor_fault(executor_fault||arbiter_error),
    .executor_error(arbiter_error?16'h0740:executor_error),.executor_error_tag,
    .cmd_data(c_cmd[143:72]),.cmd_valid(c_cmd_valid[1]),.cmd_ready(c_cmd_ready[1]),
    .status_data(c_status[15:8]),.status_valid(c_status_valid[1]),.status_ready(c_status_ready[1]),
    .dma_data(c_data[127:64]),.dma_keep(c_keep[15:8]),.dma_last(c_last[1]),.dma_valid(c_valid[1]),.dma_ready(c_ready[1]));
  tile_executor #(.P(P),.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) executor(.aclk,.aresetn,.clear_cache,
    .s_descriptor(descriptor),.s_valid(descriptor_valid&&!fault),.s_ready(descriptor_ready),
    .completion_tag,.completion_valid,.completion_ready,
    .busy(executor_busy),.fault(executor_fault),.error(executor_error),.error_tag(executor_error_tag),
    .input_hits,.parameter_hits,.overlap_cycles,
    .read_cmd_data(c_cmd[71:0]),.read_cmd_valid(c_cmd_valid[0]),.read_cmd_ready(c_cmd_ready[0]),
    .read_data(c_data[63:0]),.read_keep(c_keep[7:0]),.read_last(c_last[0]),.read_valid(c_valid[0]),.read_ready(c_ready[0]),
    .read_status_data(c_status[7:0]),.read_status_valid(c_status_valid[0]),.read_status_ready(c_status_ready[0]),
    .write_cmd_data,.write_cmd_valid,.write_cmd_ready,.write_data,.write_keep,.write_last,.write_valid,.write_ready,
    .write_status_data,.write_status_valid,.write_status_ready);
  dma_read_arbiter #(.N(2)) read_mux(.aclk,.aresetn,.c_cmd_data(c_cmd),.c_cmd_valid,.c_cmd_ready,
    .c_data,.c_keep,.c_last,.c_valid,.c_ready,.c_status,.c_status_valid,.c_status_ready,
    .cmd_data(read_cmd_data),.cmd_valid(read_cmd_valid),.cmd_ready(read_cmd_ready),
    .data(read_data),.keep(read_keep),.last(read_last),.valid(read_valid),.ready(read_ready),
    .status(read_status_data),.status_valid(read_status_valid),.status_ready(read_status_ready),
    .busy(arbiter_busy),.protocol_error(arbiter_error));
endmodule
