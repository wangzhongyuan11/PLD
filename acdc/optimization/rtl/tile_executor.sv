// Complete queued Co8 tile datapath. External interfaces are DataMover Full
// command/data/status streams; the vendor IP and AXI memory fabric are separate.
module tile_executor #(
  parameter integer P=4,
  parameter integer WORDS=(P==4?4608:3584),
  parameter integer AW=$clog2(WORDS),
  parameter integer TIMEOUT_CYCLES=10000000
) (
  input logic aclk,aresetn,clear_cache,
  input logic [511:0] s_descriptor,
  input logic s_valid,
  output logic s_ready,
  output logic [31:0] completion_tag,
  output logic completion_valid,
  input logic completion_ready,
  output logic busy,fault,
  output logic [15:0] error,
  output logic [31:0] error_tag,input_hits,parameter_hits,overlap_cycles,
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
  logic [511:0] ld,rd;
  logic islot,wslot,rislot,rwslot,input_start,parameter_start,lut_start,run_start;
  logic input_done,parameter_done,lut_done,core_done,writer_done;
  logic [7:0] input_error,parameter_error,lut_error,writer_error;
  logic [2:0] core_error;
  logic arbiter_busy,arbiter_error;
  logic [215:0] c_cmd;
  logic [2:0] c_cmd_valid,c_cmd_ready,c_last,c_valid,c_ready,c_status_valid,c_status_ready;
  logic [191:0] c_data;logic [23:0] c_keep,c_status;
  logic input_busy,parameter_busy,lut_busy,core_busy,writer_busy;
  logic load_slot,load_valid,load_ready;
  logic [$clog2(P)-1:0] load_bank;
  logic [AW-1:0] load_addr;logic [63:0] load_data;logic [7:0] load_keep;
  logic weight_slot,weight_valid,weight_ready,param_slot,param_valid,param_ready;
  logic [2:0] weight_bank;logic [7:0] weight_addr;logic [63:0] weight_data;
  logic [255:0] param_bias,param_multiplier;
  logic lut_we;logic [8:0] lut_addr;logic [7:0] lut_data;
  logic [31:0] core_data,core_tag;logic [3:0] core_keep;
  logic core_first,core_last,core_valid,core_ready;
  tile_job_queue queue(.aclk,.aresetn,.clear_cache,.s_descriptor,.s_valid,.s_ready,
    .load_descriptor(ld),.run_descriptor(rd),.load_input_slot(islot),.load_weight_slot(wslot),
    .run_input_slot(rislot),.run_weight_slot(rwslot),.input_start,.parameter_start,.lut_start,.run_start,
    .input_done,.parameter_done,.lut_done,.core_done,.writer_done,
    .input_error(arbiter_error?8'h40:input_error),.parameter_error,.lut_error,.core_error,.writer_error,
    .completion_tag,.completion_valid,.completion_ready,.busy,.fault,.error,.error_tag,
    .input_hits,.parameter_hits,.overlap_cycles);
  input_loader #(.P(P),.WORDS(WORDS),.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) inputs(
    .aclk,.aresetn,.start(input_start&&!fault),.cfg_source0(ld[31:0]),.cfg_source1(ld[63:32]),
    .cfg_image_h(ld[287:272]),.cfg_image_w(ld[271:256]),.cfg_y(ld[319:304]),.cfg_x(ld[303:288]),
    .cfg_oh(ld[351:336]),.cfg_ow(ld[335:320]),.cfg_ci(ld[359:352]),.cfg_split(ld[375:368]),
    .cfg_kernel3(ld[376]),.cfg_cat(ld[378]),.cfg_slot(islot),.lut_we,.lut_addr,.lut_data,
    .busy(input_busy),.done(input_done),.error(input_error),
    .cmd_data(c_cmd[71:0]),.cmd_valid(c_cmd_valid[0]),.cmd_ready(c_cmd_ready[0]),
    .status_data(c_status[7:0]),.status_valid(c_status_valid[0]),.status_ready(c_status_ready[0]),
    .dma_data(c_data[63:0]),.dma_keep(c_keep[7:0]),.dma_last(c_last[0]),.dma_valid(c_valid[0]),.dma_ready(c_ready[0]),
    .load_slot,.load_bank,.load_addr,.load_data,.load_keep,.load_valid,.load_ready);
  parameter_loader #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) parameters(
    .aclk,.aresetn,.start(parameter_start&&!fault),.cfg_address(ld[95:64]),
    .cfg_ci(ld[359:352]),.cfg_kernel3(ld[376]),.cfg_slot(wslot),
    .busy(parameter_busy),.done(parameter_done),.error(parameter_error),
    .cmd_data(c_cmd[143:72]),.cmd_valid(c_cmd_valid[1]),.cmd_ready(c_cmd_ready[1]),
    .status_data(c_status[15:8]),.status_valid(c_status_valid[1]),.status_ready(c_status_ready[1]),
    .dma_data(c_data[127:64]),.dma_keep(c_keep[15:8]),.dma_last(c_last[1]),.dma_valid(c_valid[1]),.dma_ready(c_ready[1]),
    .weight_slot,.weight_bank,.weight_addr,.weight_data,.weight_valid,.weight_ready,
    .param_slot,.param_valid,.param_bias,.param_multiplier,.param_ready);
  lut_loader #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) tables(
    .aclk,.aresetn,.start(lut_start&&!fault),.cfg_address(ld[127:96]),
    .busy(lut_busy),.done(lut_done),.error(lut_error),
    .cmd_data(c_cmd[215:144]),.cmd_valid(c_cmd_valid[2]),.cmd_ready(c_cmd_ready[2]),
    .status_data(c_status[23:16]),.status_valid(c_status_valid[2]),.status_ready(c_status_ready[2]),
    .dma_data(c_data[191:128]),.dma_keep(c_keep[23:16]),.dma_last(c_last[2]),.dma_valid(c_valid[2]),.dma_ready(c_ready[2]),
    .lut_we,.lut_addr,.lut_data);
  dma_read_arbiter #(.N(3)) read_mux(.aclk,.aresetn,.c_cmd_data(c_cmd),.c_cmd_valid,.c_cmd_ready,
    .c_data,.c_keep,.c_last,.c_valid,.c_ready,.c_status,.c_status_valid,.c_status_ready,
    .cmd_data(read_cmd_data),.cmd_valid(read_cmd_valid),.cmd_ready(read_cmd_ready),
    .data(read_data),.keep(read_keep),.last(read_last),.valid(read_valid),.ready(read_ready),
    .status(read_status_data),.status_valid(read_status_valid),.status_ready(read_status_ready),
    .busy(arbiter_busy),.protocol_error(arbiter_error));
  tile_conv #(.P(P),.WORDS(WORDS)) convolution(.aclk,.aresetn,.start(run_start&&!fault),
    .cfg_oh(rd[351:336]),.cfg_ow(rd[335:320]),.cfg_ci(rd[359:352]),.cfg_co(rd[363:360]),
    .cfg_kernel3(rd[376]),.cfg_relu(rd[377]),.cfg_input_slot(rislot),.cfg_weight_slot(rwslot),
    .busy(core_busy),.done(core_done),.error(core_error),
    .load_slot,.load_bank,.load_addr,.load_data,.load_keep,.load_valid,.load_ready,
    .weight_slot,.weight_bank,.weight_addr,.weight_data,.weight_valid,.weight_ready,
    .param_slot,.param_valid,.param_bias,.param_multiplier,.param_ready,
    .m_data(core_data),.m_tag(core_tag),.m_keep(core_keep),.m_first(core_first),.m_last(core_last),
    .m_valid(core_valid),.m_ready(core_ready));
  tile_writer #(.P(P),.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) writer(.aclk,.aresetn,.start(run_start&&!fault),
    .cfg_width(rd[335:320]),.cfg_height(rd[351:336]),.cfg_full_address(rd[159:128]),.cfg_full_stride(rd[191:160]),
    .cfg_pool_address(rd[223:192]),.cfg_pool_stride(rd[255:224]),.cfg_pool(rd[379]),
    .s_data(core_data),.s_tag(core_tag),.s_keep(core_keep),.s_first(core_first),.s_last(core_last),
    .s_valid(core_valid),.s_ready(core_ready),
    .cmd_data(write_cmd_data),.cmd_valid(write_cmd_valid),.cmd_ready(write_cmd_ready),
    .status_data(write_status_data),.status_valid(write_status_valid),.status_ready(write_status_ready),
    .dma_data(write_data),.dma_keep(write_keep),.dma_last(write_last),.dma_valid(write_valid),.dma_ready(write_ready),
    .busy(writer_busy),.done(writer_done),.error(writer_error));
endmodule
