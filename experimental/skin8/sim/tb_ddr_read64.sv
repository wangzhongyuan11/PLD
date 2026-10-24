`timescale 1ns/1ps
module tb_ddr_read64;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0;
  logic [31:0] cmd_addr=0,araddr;
  logic [15:0] cmd_words=0;
  logic cmd_valid=0,cmd_ready,done,error;
  logic [7:0] arlen;
  logic [2:0] arsize;
  logic [1:0] arburst;
  logic arvalid,arready=0;
  logic [63:0] rdata=0,m_data;
  logic [1:0] rresp=0;
  logic rlast=0,rvalid=0,rready,m_valid,m_last,m_ready=0;
  ddr_read64 dut(.*);
  integer pending=0,index=0,seed=6416,checked=0,seen=0,bursts=0,cycles=0,last_fault=0;
  logic [31:0] base=0;
  logic [31:0] expected_addr;
  logic [15:0] expected_words;
  logic inject_error=0;
  logic consumed=0;
  logic held_ar=0;
  logic [39:0] saved_ar;
  always @(posedge clk) begin
    cycles=cycles+1;
    consumed=rvalid && rready;
    if(resetn) begin
      if(held_ar && (!arvalid || {araddr,arlen}!==saved_ar)) $fatal(1,"AR changed under stall");
      held_ar=arvalid && !arready;saved_ar={araddr,arlen};
      if(arvalid && arready) begin
        if(pending || arsize!=3 || arburst!=1 || arlen>15 || araddr[11:0]+(arlen+1)*8>4096)
          $fatal(1,"illegal burst");
        pending=arlen+1;index=0;base=araddr;bursts=bursts+1;
        if(last_fault==1) pending=3;
        if(last_fault==2) pending=pending+1;
      end
      if(m_valid && m_ready) begin
        if(m_data!==(64'(expected_addr)+seen*8)) $fatal(1,"read data mismatch");
        if(m_last!==(seen==expected_words-1)) $fatal(1,"wrong command last");
        seen=seen+1;checked=checked+1;
      end
      if(rvalid && rready) begin index=index+1;pending=pending-1;end
    end
    if(cycles>50000) $fatal(1,"timeout");
  end
  always @(negedge clk) begin
    arready=$unsigned($random(seed))%3!=0;
    m_ready=$unsigned($random(seed))%3!=0;
    if(!rvalid || consumed) begin
      rvalid=pending>0 && $unsigned($random(seed))%4!=0;
      rdata=64'(base)+index*8;rlast=pending==1;rresp=inject_error ? 2 : 0;
    end
  end
  task automatic command(input integer address,words,expect_error);
    @(negedge clk);cmd_addr=address;cmd_words=words;cmd_valid=1;seen=0;
    expected_addr=address;expected_words=words;
    @(negedge clk);cmd_valid=0;
    // A producer may change payload immediately after handshake. Validation
    // and all subsequent bursts must use only the captured command.
    cmd_addr=32'hfffffffd;cmd_words=0;
    while(!done) @(negedge clk);
    if(error!==expect_error[0]) $fatal(1,"wrong error status");
    if(!expect_error && seen!=words) $fatal(1,"missing words");
    @(negedge clk);
  endtask
  initial begin
    repeat(4) @(negedge clk);resetn=1;
    command(32'hff0,40,0);
    command(32'h2000,1,0);
    command(32'h2f80,513,0);
    command(32'h4003,20,1);
    command(32'h4000,0,1);
    command(32'hfffffff8,2,1);
    inject_error=1;command(32'h8000,20,1);inject_error=0;
    last_fault=1;command(32'h8100,20,1);last_fault=0;
    last_fault=2;command(32'h8200,20,1);last_fault=0;
    command(32'hfffffff8,1,0);
    command(32'h9000,33,0);
    $display("PASS ddr_read64 checked=%0d bursts=%0d",checked,bursts);$finish;
  end
endmodule
