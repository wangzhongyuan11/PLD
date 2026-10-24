`timescale 1ns/1ps
module tb_dot128;
  logic clk=0; always #4 clk=~clk;
  logic resetn=0;
  logic [127:0] s_x=0;
  logic [511:0] s_w=0;
  logic [255:0] s_bias=0;
  logic [15:0] s_tag=0, m_tag;
  logic s_first=0,s_last=0,s_valid=0,s_ready;
  logic [511:0] m_acc;
  logic m_overflow,m_valid,m_ready=0;
  dot128 dut(.*);
  logic [511:0] expected [0:4095];
  logic [15:0] tags [0:4095];
  logic ovs [0:4095];
  integer wr=0,rd=0,checked=0,beats=0,cycles=0,seed=8128;
  integer signed ref_acc [0:15];
  longint signed value,product;
  logic ref_overflow=0,held=0;
  logic [528:0] held_data;
  integer first_out=-1,last_out=-1;
  logic throughput_phase=0;
  always @(posedge clk) begin
    cycles=cycles+1;
    if (!resetn) begin
      wr=0; rd=0; held=0; ref_overflow=0;
    end else begin
      if (held && (!m_valid || {m_overflow,m_tag,m_acc} !== held_data))
        $fatal(1,"output changed while stalled");
      held=m_valid && !m_ready;
      held_data={m_overflow,m_tag,m_acc};
      if (m_valid && m_ready) begin
        if (rd>=wr) $fatal(1,"unexpected or stale result");
        if (m_acc !== expected[rd] || m_tag !== tags[rd] || m_overflow !== ovs[rd])
          $fatal(1,"result mismatch index=%0d tag=%0d expected=%h actual=%h ov=%b/%b",rd,m_tag,expected[rd],m_acc,ovs[rd],m_overflow);
        rd=rd+1; checked=checked+1;
        if (throughput_phase) begin
          if (first_out<0) first_out=cycles;
          last_out=cycles;
        end
      end
      if (s_valid && s_ready) begin
        beats=beats+1;
        if (s_first) ref_overflow=0;
        for (int n=0;n<16;n++) begin
          value=s_first ? $signed(s_bias[(n%8)*32+:32]) : ref_acc[n];
          for (int i=0;i<8;i++) begin
            product=$signed(s_x[((n/8)*8+i)*8+:8]);
            product=product*$signed(s_w[((n%8)*8+i)*8+:8]);
            value=value+product;
          end
          ref_overflow=ref_overflow || value>2147483647 || value<(-2147483647-1);
          ref_acc[n]=value;
          if (s_last) expected[wr][n*32+:32]=ref_acc[n];
        end
        if (s_last) begin tags[wr]=s_tag; ovs[wr]=ref_overflow; wr=wr+1; end
      end
    end
    if (cycles>200000) $fatal(1,"timeout");
  end
  task automatic beat(input integer tag, first,last, mode, random_stalls);
    @(negedge clk);
    s_tag=tag;s_first=first;s_last=last;s_valid=1;
    for (int i=0;i<16;i++) s_x[i*8+:8]=(mode==1 ? 127 : mode==2 ? -127 : $random(seed)%128);
    for (int i=0;i<64;i++) s_w[i*8+:8]=(mode==1 || mode==2 ? 127 : $random(seed)%128);
    for (int i=0;i<8;i++) s_bias[i*32+:32]=(mode==1 ? 32'h7fffffff : mode==2 ? 32'h80000000 : $random(seed)%100000);
    m_ready=random_stalls ? ($unsigned($random(seed))%5>1) : 1;
    @(posedge clk);
    while (!s_ready) begin
      @(negedge clk);m_ready=($unsigned($random(seed))%5>1);
      @(posedge clk);
    end
  endtask
  task automatic drain;
    @(negedge clk);s_valid=0;m_ready=1;
    repeat(12) @(negedge clk);
    if (rd!=wr) $fatal(1,"missing result");
  endtask
  initial begin
    repeat(4) @(negedge clk);resetn=1;
    throughput_phase=1;
    for (int t=0;t<100;t++) beat(t,1,1,0,0);
    drain();
    if (last_out-first_out!=99) $fatal(1,"single-beat dots contain bubbles");
    throughput_phase=0;
    for (int t=100;t<600;t++) begin
      automatic integer len;
      len=1+($unsigned($random(seed))%40);
      for (int j=0;j<len;j++) begin
        beat(t,j==0,j==len-1,t%17==0?1:t%19==0?2:0,1);
        if ($unsigned($random(seed))%4==0) begin
          @(negedge clk);s_valid=0;m_ready=0;
          repeat(3) @(negedge clk);
        end
      end
    end
    drain();
    // Reset with incomplete work and blocked output; no stale result may escape.
    beat(600,1,1,1,0);
    @(negedge clk);s_valid=0;m_ready=0;
    repeat(8) @(negedge clk);
    resetn=0;
    repeat(3) @(negedge clk);
    resetn=1;
    for (int t=601;t<621;t++) beat(t,1,1,0,1);
    drain();
    if (checked!=620) $fatal(1,"wrong completed result count %0d",checked);
    $display("PASS dot128 checked=%0d accepted_beats=%0d cycles=%0d single_dot_II=1",checked,beats,cycles);
    $finish;
  end
endmodule
