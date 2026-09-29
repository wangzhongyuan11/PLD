`timescale 1ns/1ps
module tb_axi_lite_endpoint;
  logic aclk=0,aresetn=0;
  logic [11:0] s_axi_awaddr=0,s_axi_araddr=0;
  logic s_axi_awvalid=0,s_axi_awready,s_axi_wvalid=0,s_axi_wready;
  logic [31:0] s_axi_wdata=0,s_axi_rdata;
  logic [3:0] s_axi_wstrb=0;
  logic [1:0] s_axi_bresp,s_axi_rresp;
  logic s_axi_bvalid,s_axi_bready=0,s_axi_arvalid=0,s_axi_arready,s_axi_rvalid,s_axi_rready=0;
  logic wr_en,rd_en;
  logic [11:0] wr_addr,rd_addr;
  logic [31:0] wr_data,rd_data;
  logic [3:0] wr_strb;
  logic [1:0] wr_resp,rd_resp;
  logic [31:0] reg_value=0;
  int commits=0,expected_commits=0;
  axi_lite_endpoint dut(.*);
  always #5 aclk=~aclk;
  assign wr_resp=wr_addr==12'h004 ? 2'b00 : 2'b10;
  assign rd_resp=rd_addr==12'h004 ? 2'b00 : 2'b10;
  assign rd_data=rd_addr==12'h004 ? reg_value : 32'hbadadd00;
  always @(posedge aclk) begin
    if (aresetn && wr_en) begin
      commits++;
      if (wr_resp==0) for (int b=0;b<4;b++) if (wr_strb[b]) reg_value[b*8+:8]<=wr_data[b*8+:8];
    end
  end
  task automatic aw(input logic [11:0] address);
    s_axi_awaddr=address; s_axi_awvalid=1;
    do @(posedge aclk); while (!s_axi_awready);
    @(negedge aclk); s_axi_awvalid=0;
  endtask
  task automatic wd(input logic [31:0] data,input logic [3:0] strobe);
    s_axi_wdata=data; s_axi_wstrb=strobe; s_axi_wvalid=1;
    do @(posedge aclk); while (!s_axi_wready);
    @(negedge aclk); s_axi_wvalid=0;
  endtask
  task automatic response(input logic [1:0] expected);
    while (!s_axi_bvalid) @(negedge aclk);
    repeat(4) begin
      if (!s_axi_bvalid || s_axi_bresp!==expected || s_axi_awready || s_axi_wready)
        $fatal(1,"write response/backpressure violation");
      @(negedge aclk);
    end
    s_axi_bready=1; @(negedge aclk); s_axi_bready=0;
    expected_commits++;
    if (commits!=expected_commits) $fatal(1,"duplicate or missing write commit");
  endtask
  task automatic read_check(input logic [11:0] address,input logic [31:0] expected,input logic [1:0] resp);
    s_axi_araddr=address; s_axi_arvalid=1;
    do @(posedge aclk); while (!s_axi_arready);
    @(negedge aclk); s_axi_arvalid=0; s_axi_araddr=12'hffc;
    repeat(4) begin
      if (!s_axi_rvalid || s_axi_rdata!==expected || s_axi_rresp!==resp || s_axi_arready)
        $fatal(1,"read response/backpressure violation actual=%h expected=%h",s_axi_rdata,expected);
      @(negedge aclk);
    end
    s_axi_rready=1; @(negedge aclk); s_axi_rready=0;
  endtask
  initial begin
    repeat(3) @(negedge aclk); aresetn=1;
    // AW before W, W before AW, and simultaneous independent channels.
    aw(4); repeat(5) @(negedge aclk); wd(32'h12345678,4'hf); response(0);
    read_check(4,32'h12345678,0);
    wd(32'hdeadbeef,4'b0101); repeat(5) @(negedge aclk); aw(4); response(0);
    read_check(4,32'h12ad56ef,0);
    fork aw(4); wd(32'hffffffff,4'h0); join
    response(0); read_check(4,32'h12ad56ef,0);
    fork aw(12'h104); wd(32'h0,4'hf); join
    response(2); read_check(12'h104,32'hbadadd00,2);
    // A read response must remain stable while an independent write commits.
    fork
      read_check(4,32'h12ad56ef,0);
      begin aw(4); wd(32'h76543210,4'hf); response(0); end
    join
    read_check(4,32'h76543210,0);
    // Reset discards unmatched AW, so a later W cannot commit the stale address.
    aw(4); aresetn=0; repeat(3) @(negedge aclk); aresetn=1;
    wd(32'h00000000,4'hf); repeat(4) @(negedge aclk);
    if (s_axi_bvalid || commits!=expected_commits) $fatal(1,"stale address after reset");
    aresetn=0; repeat(3) @(negedge aclk); aresetn=1;
    fork aw(4); wd(32'hfeed1234,4'hf); join
    response(0); read_check(4,32'hfeed1234,0);
    $display("TB_PASS tb_axi_lite_endpoint commits=%0d",commits); $finish;
  end
  initial begin #100000; $fatal(1,"watchdog"); end
endmodule
