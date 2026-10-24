// One host command executes a DDR-resident stripe descriptor list.
// The two read masters never overlap; each engine completion drains its bus.
module frame_engine (
  input logic clk,resetn,start,
  input logic [31:0] cfg_descriptors,
  input logic [15:0] cfg_count,
  output logic busy,done,error,
  output logic [15:0] completed,
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
  logic [1023:0] descriptor;
  logic engine_start,engine_busy,engine_done,engine_error,select_engine;
  logic lut_valid,lut_ready,lut_table;
  logic [7:0] lut_index,lut_value;
  logic [31:0] cmd_addr,c_araddr,e_araddr;
  logic [15:0] cmd_words;
  logic cmd_valid,cmd_ready,rd_valid,rd_last,rd_error,rd_ready,rd_done;
  logic [63:0] rd_data;
  logic [7:0] c_arlen,e_arlen;
  logic [2:0] c_arsize,e_arsize;
  logic [1:0] c_arburst,e_arburst;
  logic c_arvalid,e_arvalid,c_rready,e_rready;
  assign select_engine=engine_busy || engine_start;
  assign araddr=select_engine ? e_araddr : c_araddr;
  assign arlen=select_engine ? e_arlen : c_arlen;
  assign arsize=select_engine ? e_arsize : c_arsize;
  assign arburst=select_engine ? e_arburst : c_arburst;
  assign arvalid=select_engine ? e_arvalid : c_arvalid;
  assign rready=select_engine ? e_rready : c_rready;
  graph_control controller(.clk,.resetn,.start,.cfg_descriptors,.cfg_count,.busy,.done,.error,.completed,
    .rd_addr(cmd_addr),.rd_words(cmd_words),.rd_cmd_valid(cmd_valid),.rd_cmd_ready(cmd_ready),
    .rd_data,.rd_valid,.rd_last,.rd_error,.rd_ready,.engine_start,.engine_busy,.engine_done,.engine_error,
    .descriptor,.lut_valid,.lut_ready,.lut_table,.lut_index,.lut_value);
  ddr_read64 control_reader(.clk,.resetn,.cmd_addr,.cmd_words,.cmd_valid,.cmd_ready,
    .done(rd_done),.error(rd_error),.araddr(c_araddr),.arlen(c_arlen),.arsize(c_arsize),.arburst(c_arburst),
    .arvalid(c_arvalid),.arready(arready && !select_engine),.rdata,.rresp,.rlast,.rvalid(rvalid && !select_engine),.rready(c_rready),
    .m_data(rd_data),.m_valid(rd_valid),.m_last(rd_last),.m_ready(rd_ready));
  stripe_engine engine(.clk,.resetn,.start(engine_start),.busy(engine_busy),.done(engine_done),.error(engine_error),
    .cfg_source(descriptor[0+:32]),.cfg_source2(descriptor[32+:32]),.cfg_weights(descriptor[64+:32]),.cfg_destination(descriptor[96+:32]),
    .cfg_width(descriptor[128+:9]),.cfg_height(descriptor[144+:9]),.cfg_y(descriptor[160+:9]),.cfg_rows(descriptor[176+:9]),
    .cfg_groups(descriptor[192+:5]),.cfg_branch_groups(descriptor[200+:5]),.cfg_mode(descriptor[208+:2]),
    .cfg_k3(descriptor[216]),.cfg_relu(descriptor[217]),.cfg_table(descriptor[218]),.cfg_table2(descriptor[219]),
    .cfg_page(descriptor[220]),.cfg_concat(descriptor[221]),.cfg_reuse_features(descriptor[222]),.cfg_feature_id(descriptor[224+:32]),
    .cfg_bias(descriptor[384+:256]),.cfg_multiplier(descriptor[640+:256]),
    .lut_valid,.lut_ready,.lut_table,.lut_index,.lut_value,
    .araddr(e_araddr),.arlen(e_arlen),.arsize(e_arsize),.arburst(e_arburst),.arvalid(e_arvalid),.arready(arready && select_engine),
    .rdata,.rresp,.rlast,.rvalid(rvalid && select_engine),.rready(e_rready),
    .awaddr,.awlen,.awsize,.awburst,.awvalid,.awready,.wdata,.wstrb,.wlast,.wvalid,.wready,.bresp,.bvalid,.bready);
endmodule
