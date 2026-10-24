`timescale 1ns/1ps
module tb_pair_to_words;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0;
  logic [127:0] s_data=0;
  logic s_second=0,s_last=0,s_valid=0,s_ready;
  logic [63:0] m_data;
  logic m_last,m_valid,m_ready=0;
  pair_to_words dut(.*);
  logic [64:0] expected [0:4095];
  logic held=0;
  logic [64:0] held_data;
  integer rd=0,wr=0,seed=1264;
  always @(posedge clk) if(resetn) begin
    if(held && (!m_valid || {m_last,m_data}!==held_data)) $fatal(1,"serializer changed while stalled");
    held=m_valid && !m_ready;held_data={m_last,m_data};
    if(m_valid && m_ready) begin
      if(rd>=wr || {m_last,m_data}!==expected[rd]) $fatal(1,"serialized word mismatch");
      rd=rd+1;
    end
    if(s_valid && s_ready) begin
      expected[wr]={s_last && !s_second,s_data[63:0]};wr=wr+1;
      if(s_second) begin expected[wr]={s_last,s_data[127:64]};wr=wr+1;end
    end
  end
  initial begin
    repeat(3) @(negedge clk);resetn=1;
    for(int n=0;n<1000;n++) begin
      @(negedge clk);s_data={$random(seed),$random(seed),$random(seed),$random(seed)};
      s_second=n%3!=0;s_last=n%7==0;s_valid=1;m_ready=$unsigned($random(seed))%3!=0;
      @(posedge clk);
      while(!s_ready) begin @(negedge clk);m_ready=$unsigned($random(seed))%3!=0;@(posedge clk);end
    end
    @(negedge clk);s_valid=0;m_ready=1;
    repeat(5) @(negedge clk);
    if(rd!=1666 || wr!=rd) $fatal(1,"missing words");
    $display("PASS pair_to_words checked=%0d",rd);$finish;
  end
endmodule
