`timescale 1ns/1ps
module tb_requant24;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0;
  logic signed [31:0] s_acc=0,s_multiplier=0;
  logic [15:0] s_tag=0,m_tag;
  logic s_relu=0,s_valid=0,s_ready;
  logic signed [7:0] m_value;
  logic m_valid,m_ready=0;
  requant24 dut(.*);
  integer expected [0:8191];
  integer tags [0:8191];
  integer wr=0,rd=0,seed=2424,cycles=0;
  longint signed p,mag,q,remn;
  logic held=0;
  logic [23:0] held_data;
  always @(posedge clk) begin
    cycles=cycles+1;
    if (!resetn) begin wr=0;rd=0;held=0;end
    else begin
      if (held && (!m_valid || {m_tag,m_value}!==held_data)) $fatal(1,"stalled quant changed");
      held=m_valid && !m_ready;held_data={m_tag,m_value};
      if (m_valid && m_ready) begin
        if (rd>=wr || $signed(m_value)!=expected[rd] || m_tag!=tags[rd])
          $fatal(1,"quant mismatch idx=%0d expected=%0d value=%0d",rd,expected[rd],m_value);
        rd=rd+1;
      end
      if (s_valid && s_ready) begin
        p=s_acc;p=p*s_multiplier;
        // Independent magnitude/division reference, not the RTL floor-bit formula.
        mag=p<0 ? -p : p;
        q=mag/16777216;remn=mag%16777216;
        if (remn>8388608 || (remn==8388608 && q%2==1)) q=q+1;
        if (p<0) q=-q;
        if (s_relu && q<0) q=0;
        if (q>127) q=127;
        if (q< -127) q=-127;
        expected[wr]=q;tags[wr]=s_tag;wr=wr+1;
      end
    end
    if (cycles>30000) $fatal(1,"timeout");
  end
  task automatic send(input integer a,b,relu);
    @(negedge clk);s_acc=a;s_multiplier=b;s_relu=relu;s_tag=s_tag+1;s_valid=1;
    m_ready=$unsigned($random(seed))%3!=0;
    @(posedge clk);
    while (!s_ready) begin
      @(negedge clk);m_ready=$unsigned($random(seed))%3!=0;
      @(posedge clk);
    end
  endtask
  initial begin
    repeat(3) @(negedge clk);resetn=1;
    for (int a=-300;a<=300;a++) begin
      send(a,8388608,0);send(a,8388608,1);
      send(a,8388607,0);send(a,8388609,0);
    end
    send(-2147483647-1,-2147483647-1,0);
    send(-2147483647-1,2147483647,0);
    send(2147483647,2147483647,0);
    send(-2147483647-1,0,0);
    for (int n=0;n<3000;n++) send($random(seed),$random(seed),n%2);
    @(negedge clk);s_valid=0;m_ready=1;
    repeat(8) @(negedge clk);
    if (rd!=5408 || wr!=rd) $fatal(1,"missing quant results %0d/%0d",rd,wr);
    $display("PASS requant24 checked=%0d",rd);$finish;
  end
endmodule
