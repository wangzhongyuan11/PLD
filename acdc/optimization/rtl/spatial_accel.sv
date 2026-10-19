// ACD8 ABI 2.0 top: AXI-Lite frame control + DataMover Full stream interfaces.
// The SoC wrapper must reset this core and BOTH DataMover directions together.
module spatial_accel #(
  parameter integer P=4,
  parameter integer CLOCK_HZ=100000000,
  parameter integer TIMEOUT_CYCLES=10000000
) (
  (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF s_axi:m_axis_read_cmd:s_axis_read_data:s_axis_read_status:m_axis_write_cmd:m_axis_write_data:s_axis_write_status, ASSOCIATED_RESET aresetn" *)
  input logic aclk,
  (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *)
  input logic aresetn,
  input logic [7:0] s_axi_awaddr,
  input logic s_axi_awvalid,
  output logic s_axi_awready,
  input logic [31:0] s_axi_wdata,
  input logic [3:0] s_axi_wstrb,
  input logic s_axi_wvalid,
  output logic s_axi_wready,
  output logic [1:0] s_axi_bresp,
  output logic s_axi_bvalid,
  input logic s_axi_bready,
  input logic [7:0] s_axi_araddr,
  input logic s_axi_arvalid,
  output logic s_axi_arready,
  output logic [31:0] s_axi_rdata,
  output logic [1:0] s_axi_rresp,
  output logic s_axi_rvalid,
  input logic s_axi_rready,
  output logic interrupt,
  output logic [71:0] m_axis_read_cmd_tdata,
  output logic m_axis_read_cmd_tvalid,
  input logic m_axis_read_cmd_tready,
  input logic [63:0] s_axis_read_data_tdata,
  input logic [7:0] s_axis_read_data_tkeep,
  input logic s_axis_read_data_tlast,s_axis_read_data_tvalid,
  output logic s_axis_read_data_tready,
  input logic [7:0] s_axis_read_status_tdata,
  input logic s_axis_read_status_tvalid,
  output logic s_axis_read_status_tready,
  output logic [71:0] m_axis_write_cmd_tdata,
  output logic m_axis_write_cmd_tvalid,
  input logic m_axis_write_cmd_tready,
  output logic [63:0] m_axis_write_data_tdata,
  output logic [7:0] m_axis_write_data_tkeep,
  output logic m_axis_write_data_tlast,m_axis_write_data_tvalid,
  input logic m_axis_write_data_tready,
  input logic [7:0] s_axis_write_status_tdata,
  input logic s_axis_write_status_tvalid,
  output logic s_axis_write_status_tready
);
  logic start,clear_cache,busy,done,fault;
  logic [31:0] cfg_program,cfg_epoch,error_tag,cycles,input_hits,parameter_hits,overlap_cycles;
  logic [15:0] cfg_count,error,submitted,completed;
  frame_control #(.P(P),.CLOCK_HZ(CLOCK_HZ)) control(.*);
  frame_engine #(.P(P),.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) engine(
    .aclk,.aresetn,.start,.clear_cache,.cfg_program,.cfg_epoch,.cfg_count,
    .busy,.done,.fault,.error,.error_tag,.cycles,.submitted,.completed,.input_hits,.parameter_hits,.overlap_cycles,
    .read_cmd_data(m_axis_read_cmd_tdata),.read_cmd_valid(m_axis_read_cmd_tvalid),.read_cmd_ready(m_axis_read_cmd_tready),
    .read_data(s_axis_read_data_tdata),.read_keep(s_axis_read_data_tkeep),.read_last(s_axis_read_data_tlast),
    .read_valid(s_axis_read_data_tvalid),.read_ready(s_axis_read_data_tready),
    .read_status_data(s_axis_read_status_tdata),.read_status_valid(s_axis_read_status_tvalid),.read_status_ready(s_axis_read_status_tready),
    .write_cmd_data(m_axis_write_cmd_tdata),.write_cmd_valid(m_axis_write_cmd_tvalid),.write_cmd_ready(m_axis_write_cmd_tready),
    .write_data(m_axis_write_data_tdata),.write_keep(m_axis_write_data_tkeep),.write_last(m_axis_write_data_tlast),
    .write_valid(m_axis_write_data_tvalid),.write_ready(m_axis_write_data_tready),
    .write_status_data(s_axis_write_status_tdata),.write_status_valid(s_axis_write_status_tvalid),.write_status_ready(s_axis_write_status_tready));
endmodule
