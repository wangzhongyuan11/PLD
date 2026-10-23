`timescale 1ns/1ps
module tb_frame_control;
  logic aclk=0,aresetn=0;
  logic [7:0] s_axi_awaddr=0,s_axi_araddr=0;
  logic s_axi_awvalid=0,s_axi_awready,s_axi_wvalid=0,s_axi_wready,s_axi_bvalid,s_axi_bready=0;
  logic s_axi_arvalid=0,s_axi_arready,s_axi_rvalid,s_axi_rready=0;
  logic [31:0] s_axi_wdata=0,s_axi_rdata;logic [3:0] s_axi_wstrb=0;
  logic [1:0] s_axi_bresp,s_axi_rresp;
  logic start,clear_cache,interrupt;logic [31:0] cfg_program,cfg_epoch;logic [15:0] cfg_count;
  logic busy=0,done=0,fault=0;
  logic [15:0] error=0,submitted=3,completed=2;
  logic [31:0] error_tag=17,cycles=0,input_hits=12,parameter_hits=13,overlap_cycles=14;
  int timer=0,starts=0,clears=0,writes=0,reads=0,watchdog=0;
  logic [31:0] value,held_read;logic [1:0] held_rresp,held_bresp;bit rstall=0,bstall=0;
  frame_control dut(.*);
  always #5 aclk=~aclk;
  always @(posedge aclk)if(aresetn)begin
    cycles<=cycles+1;done<=0;watchdog++;
    if(start)begin starts++;busy<=1;timer=150;end
    else if(timer>0)begin timer--;if(timer==0)begin busy<=0;done<=1;end end
    if(clear_cache)clears++;
    if(rstall&&(!s_axi_rvalid||{s_axi_rdata,s_axi_rresp}!={held_read,held_rresp}))$fatal(1,"AXI read response changed under backpressure");
    if(bstall&&(!s_axi_bvalid||s_axi_bresp!=held_bresp))$fatal(1,"AXI write response changed under backpressure");
    rstall=s_axi_rvalid&&!s_axi_rready;held_read=s_axi_rdata;held_rresp=s_axi_rresp;
    bstall=s_axi_bvalid&&!s_axi_bready;held_bresp=s_axi_bresp;
    if(watchdog>10000)$fatal(1,"AXI control deadlock");
  end else begin rstall=0;bstall=0;end
  task automatic aw(input logic [7:0] addr);
    @(negedge aclk);s_axi_awaddr=addr;s_axi_awvalid=1;
    do @(posedge aclk);while(!s_axi_awready);
    @(negedge aclk);s_axi_awvalid=0;
  endtask
  task automatic w(input logic [31:0] data,input logic [3:0] mask);
    @(negedge aclk);s_axi_wdata=data;s_axi_wstrb=mask;s_axi_wvalid=1;
    do @(posedge aclk);while(!s_axi_wready);
    @(negedge aclk);s_axi_wvalid=0;
  endtask
  task automatic write_reg(input logic [7:0] addr,input logic [31:0] data,input logic [3:0] mask,input logic [1:0] expected,input bit data_first);
    if(data_first)begin w(data,mask);repeat(3)@(negedge aclk);aw(addr);end
    else begin aw(addr);repeat(3)@(negedge aclk);w(data,mask);end
    wait(s_axi_bvalid);repeat(4)@(negedge aclk);
    if(s_axi_bresp!=expected)$fatal(1,"wrong AXI write response addr=%h resp=%0d expected=%0d",addr,s_axi_bresp,expected);
    s_axi_bready=1;@(negedge aclk);s_axi_bready=0;writes++;
  endtask
  task automatic read_reg(input logic [7:0] addr,input logic [1:0] expected);
    @(negedge aclk);s_axi_araddr=addr;s_axi_arvalid=1;
    do @(posedge aclk);while(!s_axi_arready);
    @(negedge aclk);s_axi_arvalid=0;
    wait(s_axi_rvalid);value=s_axi_rdata;repeat(4)@(negedge aclk);
    if(s_axi_rresp!=expected)$fatal(1,"wrong AXI read response");
    s_axi_rready=1;@(negedge aclk);s_axi_rready=0;reads++;
  endtask
  initial begin
    repeat(5)@(negedge aclk);aresetn=1;
    read_reg(0,0);if(value!=32'h41434438)$fatal(1,"magic");
    read_reg(4,0);if(value!=32'h00020000)$fatal(1,"ABI");
    read_reg(8,0);if(value!=32'h001f0804)$fatal(1,"capabilities");
    write_reg(8'h14,32'h10800000,15,0,0);
    write_reg(8'h18,1040,15,0,1);
    write_reg(8'h1c,32'h12345678,15,0,1);
    write_reg(8'h1c,32'h0000aa00,2,0,0);
    read_reg(8'h1c,0);if(value!=32'h1234aa78)$fatal(1,"byte mask clobbered neighboring bytes");
    write_reg(0,1,15,2,0);write_reg(8'h15,0,15,2,1);
    write_reg(8'h18,32'h00010001,15,2,1);write_reg(8'h0c,3,15,2,0);
    write_reg(8'h0c,2,15,0,1);
    write_reg(8'h0c,1,15,0,0);
    if(!busy||starts!=1||cfg_program!=32'h10800000||cfg_count!=1040||clears!=1)$fatal(1,"frame start/configuration");
    write_reg(8'h14,0,15,2,1);write_reg(8'h0c,1,15,2,0);write_reg(8'h0c,2,15,2,1);
    read_reg(8'h10,0);if(!value[0])$fatal(1,"missing busy");
    wait(done);repeat(3)@(negedge aclk);
    read_reg(8'h10,0);if(value[0]||!value[1]||!interrupt)$fatal(1,"missing sticky done");
    read_reg(8'h2c,0);if(value!=32'h00020003)$fatal(1,"progress counters");
    read_reg(8'h3c,0);if(value!=100000000)$fatal(1,"clock metadata");
    read_reg(8'h40,2);read_reg(8'h01,2);
    write_reg(8'h0c,4,15,0,1);if(interrupt)$fatal(1,"ACK_DONE did not clear notification");
    fault=1;error=16'h0288;
    write_reg(8'h0c,1,15,2,0);write_reg(8'h1c,4,15,2,1);
    read_reg(8'h20,0);if(value!=16'h0288||!interrupt)$fatal(1,"fault visibility");
    $display("TB_PASS frame_control writes=%0d reads=%0d AW_W_independent=PASS byte_strobes=PASS backpressure=PASS busy_lock=PASS",writes,reads);$finish;
  end
endmodule
