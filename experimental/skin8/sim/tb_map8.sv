`timescale 1ns/1ps
module tb_map8;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0,cfg_lock=0,cfg_table=0,cfg_valid=0,cfg_ready;
  logic [7:0] cfg_index=0,cfg_value=0;
  logic [63:0] s_data=0,m_data;
  logic s_table=0,s_valid=0,s_ready,m_valid,m_ready=0;
  map8 dut(.*);
  logic [63:0] expected [0:2047];
  integer wr=0,rd=0,seed=8256;
  logic held=0;
  logic [63:0] held_data;
  function automatic logic [7:0] table_value(input integer bank,v);
    // Synthetic tables exercise all 256 indices, including negative byte codes.
    return bank ? (v*37+17)%256 : (v*13+29)%256;
  endfunction
  always @(posedge clk) if(resetn) begin
    if(cfg_lock && cfg_ready) $fatal(1,"locked LUT accepted write");
    if(held && (!m_valid || m_data!==held_data)) $fatal(1,"stalled LUT output changed");
    held=m_valid && !m_ready;held_data=m_data;
    if(m_valid && m_ready) begin
      if(rd>=wr || m_data!==expected[rd]) $fatal(1,"LUT result mismatch %0d",rd);
      rd=rd+1;
    end
    if(s_valid && s_ready) begin
      for(int lane=0;lane<8;lane++) expected[wr][lane*8+:8]=table_value(s_table,s_data[lane*8+:8]);
      wr=wr+1;
    end
  end
  initial begin
    repeat(3) @(negedge clk);resetn=1;
    for(int t=0;t<2;t++) for(int v=0;v<256;v++) begin
      @(negedge clk);cfg_valid=1;cfg_table=t;cfg_index=v;cfg_value=table_value(t,v);
    end
    @(negedge clk);cfg_valid=0;cfg_lock=1;
    for(int n=0;n<1024;n++) begin
      @(negedge clk);s_valid=1;s_table=n/256%2;
      for(int lane=0;lane<8;lane++) s_data[lane*8+:8]=(n+lane*31)%256;
      cfg_valid=1;cfg_value=0;cfg_index=n;cfg_table=s_table;
      m_ready=$unsigned($random(seed))%3!=0;
      @(posedge clk);
      while(!s_ready) begin @(negedge clk);m_ready=$unsigned($random(seed))%3!=0;@(posedge clk);end
    end
    @(negedge clk);s_valid=0;cfg_valid=0;m_ready=1;
    repeat(5) @(negedge clk);
    if(rd!=1024 || wr!=rd) $fatal(1,"missing lookups");
    $display("PASS map8 checked=%0d all_byte_indices=1 tables=2",rd*8);$finish;
  end
endmodule
