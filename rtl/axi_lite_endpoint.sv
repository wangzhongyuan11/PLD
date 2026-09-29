// One outstanding read and write, independent AW/W capture. Backend sees a
// one-cycle write commit, and supplies combinational response/read data.
module axi_lite_endpoint #(
  parameter int ADDR_WIDTH=12
) (
  input logic aclk, aresetn,
  input logic [ADDR_WIDTH-1:0] s_axi_awaddr,
  input logic s_axi_awvalid,
  output logic s_axi_awready,
  input logic [31:0] s_axi_wdata,
  input logic [3:0] s_axi_wstrb,
  input logic s_axi_wvalid,
  output logic s_axi_wready,
  output logic [1:0] s_axi_bresp,
  output logic s_axi_bvalid,
  input logic s_axi_bready,
  input logic [ADDR_WIDTH-1:0] s_axi_araddr,
  input logic s_axi_arvalid,
  output logic s_axi_arready,
  output logic [31:0] s_axi_rdata,
  output logic [1:0] s_axi_rresp,
  output logic s_axi_rvalid,
  input logic s_axi_rready,
  output logic wr_en,
  output logic [ADDR_WIDTH-1:0] wr_addr,
  output logic [31:0] wr_data,
  output logic [3:0] wr_strb,
  input logic [1:0] wr_resp,
  output logic rd_en,
  output logic [ADDR_WIDTH-1:0] rd_addr,
  input logic [31:0] rd_data,
  input logic [1:0] rd_resp
);
  logic aw_pending,w_pending;
  assign s_axi_awready=aresetn && !aw_pending && !s_axi_bvalid;
  assign s_axi_wready=aresetn && !w_pending && !s_axi_bvalid;
  assign wr_en=aresetn && aw_pending && w_pending && !s_axi_bvalid;
  assign s_axi_arready=aresetn && !s_axi_rvalid;
  assign rd_en=s_axi_arvalid && s_axi_arready;
  assign rd_addr=s_axi_araddr;
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      aw_pending<=0; w_pending<=0; s_axi_bvalid<=0; s_axi_rvalid<=0;
    end else begin
      if (s_axi_awvalid && s_axi_awready) begin
        wr_addr<=s_axi_awaddr; aw_pending<=1;
      end
      if (s_axi_wvalid && s_axi_wready) begin
        wr_data<=s_axi_wdata; wr_strb<=s_axi_wstrb; w_pending<=1;
      end
      if (wr_en) begin
        aw_pending<=0; w_pending<=0;
        s_axi_bvalid<=1; s_axi_bresp<=wr_resp;
      end else if (s_axi_bvalid && s_axi_bready) s_axi_bvalid<=0;
      if (rd_en) begin
        s_axi_rdata<=rd_data; s_axi_rresp<=rd_resp; s_axi_rvalid<=1;
      end else if (s_axi_rvalid && s_axi_rready) s_axi_rvalid<=0;
    end
  end
endmodule
