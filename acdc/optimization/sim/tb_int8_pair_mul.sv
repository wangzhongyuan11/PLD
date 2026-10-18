`timescale 1ns/1ps
module tb_int8_pair_mul;
  logic aclk=0,aresetn=0;
  logic signed [7:0] s_a0=0,s_a1=0,s_weight=0;
  logic s_valid=0,s_ready,m_valid,m_ready=0;
  logic signed [15:0] m_p0,m_p1;
  logic [31:0] expected[$],word;
  logic [31:0] held;
  bit was_stalled=0;
  int accepted=0,received=0;
  int unsigned rng=32'h2516acdc;
  int edges[7]='{-128,-127,-1,0,1,126,127};
  int8_pair_mul dut(.*);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(posedge aclk) begin
    if (!aresetn) begin expected.delete();was_stalled=0;end
    else begin
      if (was_stalled && (!m_valid || {m_p1,m_p0}!==held))
        $fatal(1,"output changed under backpressure");
      was_stalled=m_valid&&!m_ready;held={m_p1,m_p0};
      if(s_valid&&s_ready)begin
        word={16'($signed(s_a1)*$signed(s_weight)),16'($signed(s_a0)*$signed(s_weight))};
        expected.push_back(word);accepted++;
      end
      if(m_valid&&m_ready)begin
        if(!expected.size())$fatal(1,"unexpected output");
        word=expected.pop_front();
        if({m_p1,m_p0}!==word)$fatal(1,"pair mismatch index=%0d actual=%h expected=%h",received,{m_p1,m_p0},word);
        received++;
      end
    end
  end
  task automatic send(input int a,input int b,input int w);
    @(negedge aclk);
    s_a0=8'(a);s_a1=8'(b);s_weight=8'(w);s_valid=1;
    do @(posedge aclk);while(!s_ready);
  endtask
  always @(negedge aclk)begin
    rng=next_rng(rng);m_ready=aresetn&&(rng[1]||rng[5]);
  end
  initial begin
    repeat(4)@(negedge aclk);aresetn=1;
    for(int a=0;a<7;a++)for(int b=0;b<7;b++)for(int w=0;w<7;w++)send(edges[a],edges[b],edges[w]);
    for(int j=0;j<20000;j++)begin
      rng=next_rng(rng);send(int'(rng[7:0])-128,int'(rng[15:8])-128,int'(rng[23:16])-128);
      if(j%7==0)begin @(negedge aclk);s_valid=0;repeat(j%3+1)@(negedge aclk);end
    end
    @(negedge aclk);s_valid=0;
    wait(received==accepted);repeat(5)@(negedge aclk);
    // Reset while transactions are in flight; old results must disappear.
    send(-128,127,-127);send(127,-128,127);
    @(negedge aclk);aresetn=0;s_valid=0;
    repeat(3)@(negedge aclk);accepted=0;received=0;aresetn=1;
    send(-127,127,-127);send(127,-127,127);
    @(negedge aclk);s_valid=0;wait(received==accepted);
    repeat(8)@(negedge aclk);
    $display("TB_PASS int8_pair_mul directed=343 random=20000 reset_recovery=%0d",received);$finish;
  end
  initial begin #2000000;$fatal(1,"pair watchdog");end
endmodule
