`timescale 1ns/1ps
module tb_ddr_write64;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0;
  logic [31:0] cmd_addr=0,awaddr;
  logic [15:0] cmd_words=0;
  logic cmd_valid=0,cmd_ready,done,error;
  logic [63:0] s_data=0,wdata;
  logic [7:0] s_strb=0,wstrb,awlen;
  logic s_valid=0,s_ready;
  logic [2:0] awsize;
  logic [1:0] awburst,bresp=0;
  logic awvalid,awready=0,wlast,wvalid,wready=0,bvalid=0,bready;
  ddr_write64 dut(.*);
  integer pending=0,seen=0,checked=0,bursts=0,seed=1664,cycles=0;
  logic need_response=0,inject_error=0,consumed=0,consumed_b=0;
  logic held_aw=0,held_w=0;
  logic [39:0] saved_aw;
  logic [72:0] saved_w;
  always @(posedge clk) begin
    cycles=cycles+1;consumed=s_valid && s_ready;consumed_b=bvalid && bready;
    if(resetn) begin
      if(held_aw && (!awvalid || {awaddr,awlen}!==saved_aw)) $fatal(1,"AW unstable");
      if(held_w && (!wvalid || {wdata,wstrb,wlast}!==saved_w)) $fatal(1,"W unstable");
      held_aw=awvalid && !awready;saved_aw={awaddr,awlen};
      held_w=wvalid && !wready;saved_w={wdata,wstrb,wlast};
      if(awvalid && awready) begin
        if(pending || need_response || awsize!=3 || awburst!=1 || awlen>15 || awaddr[11:0]+(awlen+1)*8>4096)
          $fatal(1,"illegal write burst");
        if(awaddr!==cmd_addr+seen*8) $fatal(1,"wrong write address");
        pending=awlen+1;bursts=bursts+1;
      end
      if(wvalid && wready) begin
        if(!pending || wdata!==(64'(cmd_addr)+seen*8) || wstrb!==(8'(seen)^8'ha5) || wlast!==(pending==1))
          $fatal(1,"write mismatch seen=%0d",seen);
        pending=pending-1;seen=seen+1;checked=checked+1;
        if(pending==0) need_response=1;
      end
      if(bvalid && bready) need_response=0;
    end
    if(cycles>50000) $fatal(1,"timeout");
  end
  always @(negedge clk) begin
    awready=$unsigned($random(seed))%3!=0;
    wready=$unsigned($random(seed))%3!=0;
    if(!bvalid || consumed_b) begin
      bvalid=need_response && $unsigned($random(seed))%4!=0;bresp=inject_error?2:0;
    end
    if(!s_valid || consumed) begin
      s_valid=seen<cmd_words && $unsigned($random(seed))%4!=0;
      s_data=64'(cmd_addr)+seen*8;s_strb=8'(seen)^8'ha5;
    end
  end
  task automatic command(input integer address,words,expect_error);
    @(negedge clk);cmd_addr=address;cmd_words=words;cmd_valid=1;seen=0;s_valid=0;
    @(negedge clk);cmd_valid=0;
    while(!done) @(negedge clk);
    if(error!==expect_error[0]) $fatal(1,"wrong write status");
    if(!expect_error && seen!=words) $fatal(1,"missing write beats");
    cmd_words=0;s_valid=0;
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
    command(32'hfffffff8,1,0);
    command(32'h9000,33,0);
    $display("PASS ddr_write64 checked=%0d bursts=%0d",checked,bursts);$finish;
  end
endmodule
