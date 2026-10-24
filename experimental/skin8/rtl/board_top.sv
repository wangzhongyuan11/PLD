// PS AXI-Lite control and 64-bit DDR AXI master wrapper.
module board_top (
  (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF S_AXI:M_AXI, ASSOCIATED_RESET resetn, FREQ_HZ 125000000" *) input logic clk,
  (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 resetn RST", X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *) input logic resetn,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWADDR", X_INTERFACE_PARAMETER="PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 8, FREQ_HZ 125000000, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1" *) input logic [7:0] s_axi_awaddr,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *) input logic [2:0] s_axi_awprot,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input logic s_axi_awvalid,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output logic s_axi_awready,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WDATA" *) input logic [31:0] s_axi_wdata,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *) input logic [3:0] s_axi_wstrb,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WVALID" *) input logic s_axi_wvalid,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WREADY" *) output logic s_axi_wready,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BRESP" *) output logic [1:0] s_axi_bresp,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BVALID" *) output logic s_axi_bvalid,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BREADY" *) input logic s_axi_bready,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *) input logic [7:0] s_axi_araddr,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *) input logic [2:0] s_axi_arprot,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input logic s_axi_arvalid,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output logic s_axi_arready,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RDATA" *) output logic [31:0] s_axi_rdata,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RRESP" *) output logic [1:0] s_axi_rresp,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RVALID" *) output logic s_axi_rvalid,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RREADY" *) input logic s_axi_rready,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARADDR", X_INTERFACE_PARAMETER="PROTOCOL AXI4, DATA_WIDTH 64, ADDR_WIDTH 32, FREQ_HZ 125000000, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1" *) output logic [31:0] m_axi_araddr,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARLEN" *) output logic [7:0] m_axi_arlen,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARSIZE" *) output logic [2:0] m_axi_arsize,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARBURST" *) output logic [1:0] m_axi_arburst,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARLOCK" *) output logic m_axi_arlock,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARCACHE" *) output logic [3:0] m_axi_arcache,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARPROT" *) output logic [2:0] m_axi_arprot,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARQOS" *) output logic [3:0] m_axi_arqos,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARVALID" *) output logic m_axi_arvalid,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARREADY" *) input logic m_axi_arready,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RDATA" *) input logic [63:0] m_axi_rdata,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RRESP" *) input logic [1:0] m_axi_rresp,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RLAST" *) input logic m_axi_rlast,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RVALID" *) input logic m_axi_rvalid,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RREADY" *) output logic m_axi_rready,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI AWADDR" *) output logic [31:0] m_axi_awaddr,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI AWLEN" *) output logic [7:0] m_axi_awlen,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI AWSIZE" *) output logic [2:0] m_axi_awsize,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI AWBURST" *) output logic [1:0] m_axi_awburst,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI AWLOCK" *) output logic m_axi_awlock,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI AWCACHE" *) output logic [3:0] m_axi_awcache,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI AWPROT" *) output logic [2:0] m_axi_awprot,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI AWQOS" *) output logic [3:0] m_axi_awqos,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI AWVALID" *) output logic m_axi_awvalid,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI AWREADY" *) input logic m_axi_awready,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI WDATA" *) output logic [63:0] m_axi_wdata,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI WSTRB" *) output logic [7:0] m_axi_wstrb,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI WLAST" *) output logic m_axi_wlast,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI WVALID" *) output logic m_axi_wvalid,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI WREADY" *) input logic m_axi_wready,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI BRESP" *) input logic [1:0] m_axi_bresp,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI BVALID" *) input logic m_axi_bvalid,
  (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI BREADY" *) output logic m_axi_bready
);
  logic start,busy,done,error;
  logic [31:0] cfg_descriptors;
  logic [15:0] cfg_count,completed;
  assign m_axi_arlock=0;assign m_axi_awlock=0;
  assign m_axi_arcache=4'b0011;assign m_axi_awcache=4'b0011;
  assign m_axi_arprot=0;assign m_axi_awprot=0;assign m_axi_arqos=0;assign m_axi_awqos=0;
  frame_registers registers(.clk,.resetn,.start,.busy,.done,.error,.cfg_descriptors,.cfg_count,.completed,
    .s_awaddr(s_axi_awaddr),.s_awvalid(s_axi_awvalid),.s_awready(s_axi_awready),.s_wdata(s_axi_wdata),.s_wstrb(s_axi_wstrb),.s_wvalid(s_axi_wvalid),.s_wready(s_axi_wready),.s_bresp(s_axi_bresp),.s_bvalid(s_axi_bvalid),.s_bready(s_axi_bready),.s_araddr(s_axi_araddr),.s_arvalid(s_axi_arvalid),.s_arready(s_axi_arready),.s_rdata(s_axi_rdata),.s_rresp(s_axi_rresp),.s_rvalid(s_axi_rvalid),.s_rready(s_axi_rready));
  frame_engine engine(.clk,.resetn,.start,.busy,.done,.error,.cfg_descriptors,.cfg_count,.completed,
    .araddr(m_axi_araddr),.arlen(m_axi_arlen),.arsize(m_axi_arsize),.arburst(m_axi_arburst),.arvalid(m_axi_arvalid),.arready(m_axi_arready),.rdata(m_axi_rdata),.rresp(m_axi_rresp),.rlast(m_axi_rlast),.rvalid(m_axi_rvalid),.rready(m_axi_rready),.awaddr(m_axi_awaddr),.awlen(m_axi_awlen),.awsize(m_axi_awsize),.awburst(m_axi_awburst),.awvalid(m_axi_awvalid),.awready(m_axi_awready),.wdata(m_axi_wdata),.wstrb(m_axi_wstrb),.wlast(m_axi_wlast),.wvalid(m_axi_wvalid),.wready(m_axi_wready),.bresp(m_axi_bresp),.bvalid(m_axi_bvalid),.bready(m_axi_bready));
endmodule
