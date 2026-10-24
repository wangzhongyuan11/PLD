`timescale 1ns/1ps
module tb_nearest2_row;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0,start=0,cfg_replay=0;
  logic [8:0] cfg_width=0;
  logic busy,done,error;
  logic [63:0] s_data=0,m_data;
  logic s_valid=0,s_ready,m_valid,m_last,m_ready=0;
  nearest2_row dut(.*);
  integer seed=2222,seen=0,checked=0,epoch=0;
  logic held=0;
  logic [64:0] held_data;
  function automatic logic [63:0] pixel(input integer x);
    return 64'h8123456789abcdef^(64'(x)*64'h1234567)^(64'(epoch)<<32);
  endfunction
  always @(posedge clk) if(resetn) begin
    if(held && (!m_valid || {m_last,m_data}!==held_data)) $fatal(1,"nearest output unstable");
    held=m_valid && !m_ready;held_data={m_last,m_data};
    if(m_valid && m_ready) begin
      if(m_data!==pixel(seen/2) || m_last!==(seen==cfg_width*2-1)) $fatal(1,"nearest mismatch idx=%0d",seen);
      seen=seen+1;checked=checked+1;
    end
  end
  task automatic run_case(input integer width,replay);
    @(negedge clk);cfg_width=width;cfg_replay=replay;start=1;seen=0;
    if(!replay) epoch=epoch+1;
    @(negedge clk);start=0;
    if(!replay) for(int n=0;n<width;n++) begin
      @(negedge clk);s_valid=1;s_data=pixel(n);m_ready=$unsigned($random(seed))%3!=0;
      @(posedge clk);
      while(!s_ready) begin @(negedge clk);m_ready=$unsigned($random(seed))%3!=0;@(posedge clk);end
    end
    @(negedge clk);s_valid=0;
    while(busy) begin m_ready=$unsigned($random(seed))%3!=0;@(negedge clk);end
    if(error || seen!=width*2) $fatal(1,"nearest missing output");
  endtask
  initial begin
    repeat(3) @(negedge clk);resetn=1;
    cfg_replay=1;cfg_width=8;start=1;
    @(negedge clk);start=0;
    if(!error || busy) $fatal(1,"uninitialized replay accepted");
    run_case(1,0);run_case(1,1);
    run_case(7,0);run_case(7,1);
    run_case(128,0);run_case(128,1);
    run_case(256,0);run_case(256,1);
    run_case(256,0);run_case(256,1);
    $display("PASS nearest2_row checked=%0d",checked);$finish;
  end
endmodule
