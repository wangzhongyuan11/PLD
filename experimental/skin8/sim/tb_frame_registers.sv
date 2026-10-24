`timescale 1ns/1ps
module tb_frame_registers;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0;
  logic [7:0] s_awaddr=0,s_araddr=0;
  logic s_awvalid=0,s_awready,s_wvalid=0,s_wready,s_bvalid,s_bready=0;
  logic [31:0] s_wdata=0,s_rdata,cfg_descriptors;
  logic [3:0] s_wstrb=0;
  logic [1:0] s_bresp,s_rresp;
  logic s_arvalid=0,s_arready,s_rvalid,s_rready=0,start,busy=0,done=0,error=0;
  logic [15:0] cfg_count,completed=0;
  integer checked=0,starts=0,cycles=0;
  frame_registers dut(.*);
  always @(posedge clk) begin
    cycles=cycles+1;if(start) starts=starts+1;
    if(cycles>10000) $fatal(1,"register test timeout");
  end
  task automatic write_reg(input integer a,v,mask,skew,expected_response);
    fork
      begin
        repeat(skew>0 ? skew : 0) @(negedge clk);
        @(negedge clk);s_awaddr=a;s_awvalid=1;
        do @(posedge clk); while(!s_awready);
        @(negedge clk);s_awvalid=0;
      end
      begin
        repeat(skew<0 ? -skew : 0) @(negedge clk);
        @(negedge clk);s_wdata=v;s_wstrb=mask;s_wvalid=1;
        do @(posedge clk); while(!s_wready);
        @(negedge clk);s_wvalid=0;
      end
    join
    while(!s_bvalid) @(negedge clk);
    repeat(4) begin
      @(negedge clk);
      if(!s_bvalid || s_bresp!=expected_response) $fatal(1,"unstable/incorrect write response");
    end
    s_bready=1;@(negedge clk);s_bready=0;checked=checked+1;
  endtask
  task automatic read_reg(input integer a,expected,expected_response);
    logic [31:0] hold_data;
    @(negedge clk);s_araddr=a;s_arvalid=1;
    do @(posedge clk); while(!s_arready);
    @(negedge clk);s_arvalid=0;
    while(!s_rvalid) @(negedge clk);
    hold_data=s_rdata;
    repeat(4) begin
      @(negedge clk);
      if(!s_rvalid || s_rresp!=expected_response || s_rdata!==hold_data || s_rdata!==expected)
        $fatal(1,"incorrect read address=%h expected=%h got=%h",a,expected,s_rdata);
    end
    s_rready=1;@(negedge clk);s_rready=0;checked=checked+1;
  endtask
  initial begin
    repeat(4) @(negedge clk);resetn=1;
    read_reg(4,32'h534b3800,0);
    write_reg(16,32'h12345680,15,7,0);read_reg(16,32'h12345680,0);
    write_reg(16,32'hff00aa00,5,-9,0);read_reg(16,32'h12005600,0);
    write_reg(20,32'hffffffff,1,0,0);read_reg(20,255,0);
    write_reg(20,32'h00000300,2,-3,0);read_reg(20,1023,0);
    write_reg(0,1,0,2,0);if(starts) $fatal(1,"WSTRB ignored for start");
    write_reg(0,1,1,-2,0);if(starts!=1) $fatal(1,"missing start pulse");
    @(negedge clk);busy=1;
    read_reg(0,2,0);write_reg(16,0,15,4,2);write_reg(0,1,1,-4,2);
    if(starts!=1) $fatal(1,"busy start accepted");
    @(negedge clk);busy=0;done=1;error=1;completed=12;
    @(negedge clk);done=0;
    read_reg(0,12,0);read_reg(28,12,0);read_reg(16,32'h12005600,0);
    write_reg(0,1,1,0,0);read_reg(0,0,0);
    write_reg(4,0,15,0,2);read_reg(3,0,2);read_reg(32,0,2);
    $display("PASS frame_registers checked=%0d split_aw_w_and_backpressure=1",checked);$finish;
  end
endmodule
