`timescale 1ns/1ps
module tb_conv_array # (parameter integer P=4);
  logic aclk=0,aresetn=0;
  logic [P*64-1:0] s_input=0;
  logic [511:0] s_weights=0;
  logic [255:0] s_bias=0;
  logic [31:0] s_tag=0,m_tag;
  logic [P*256-1:0] m_acc;
  logic s_first=0,s_last=0,s_valid=0,s_ready,m_valid,m_ready=0,m_overflow;
  logic [P*256+32:0] expected[$],item;
  longint signed sums[P][8],term;
  bit overflow,stalled=0;
  logic [P*256+32:0] held;
  int sent=0,received=0;
  int unsigned rng=32'hacdc2564;
  conv_array #(.P(P)) dut(.*);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk)begin rng=next_rng(rng);m_ready=aresetn&&(rng[1]||rng[4]);end
  always @(posedge aclk)begin
    if(!aresetn)begin expected.delete();stalled=0;end
    else begin
      if(stalled&&(!m_valid||{m_tag,m_overflow,m_acc}!==held))$fatal(1,"array output unstable under stall");
      stalled=m_valid&&!m_ready;held={m_tag,m_overflow,m_acc};
      if(m_valid&&m_ready)begin
        if(!expected.size())$fatal(1,"unexpected array result");
        item=expected.pop_front();
        if({m_tag,m_overflow,m_acc}!==item)$fatal(1,"array mismatch P=%0d result=%0d lane0=%h/%h overflow=%b/%b",P,received,m_acc[31:0],item[31:0],m_overflow,item[P*256]);
        received++;
      end
    end
  end
  task automatic send(input int count,input bit inject_overflow,input bit gaps);
    logic [P*256+32:0] e;
    overflow=0;
    for(int tap=0;tap<count;tap++)begin
      @(negedge aclk);s_valid=0;
      if(gaps)repeat(rng%3)@(negedge aclk);
      for(int p=0;p<P;p++)for(int i=0;i<8;i++)begin
        rng=next_rng(rng);s_input[(p*8+i)*8+:8]=inject_overflow?8'd127:8'(rng);
      end
      for(int o=0;o<8;o++)begin
        rng=next_rng(rng);s_bias[o*32+:32]=inject_overflow?32'h7fffffff:int'(rng%65536)-32768;
        for(int i=0;i<8;i++)begin rng=next_rng(rng);s_weights[(o*8+i)*8+:8]=inject_overflow?8'd127:8'(rng);end
      end
      for(int p=0;p<P;p++)for(int o=0;o<8;o++)begin
        if(tap==0)sums[p][o]=$signed(s_bias[o*32+:32]);
        term=0;
        for(int i=0;i<8;i++)term+=$signed(s_input[(p*8+i)*8+:8])*$signed(s_weights[(o*8+i)*8+:8]);
        sums[p][o]+=term;
        if(sums[p][o]>64'sd2147483647||sums[p][o]<-64'sd2147483648)overflow=1;
        sums[p][o]=$signed(32'(sums[p][o]));
      end
      s_first=tap==0;s_last=tap==count-1;s_valid=1;s_tag=32'(sent)^32'hacdc2500;
      do @(posedge aclk);while(!s_ready);
      if(s_last)begin
        for(int p=0;p<P;p++)for(int o=0;o<8;o++)e[(p*8+o)*32+:32]=32'(sums[p][o]);
        e[P*256]=overflow;e[P*256+1+:32]=s_tag;expected.push_back(e);sent++;
      end
    end
    @(negedge aclk);s_valid=0;
  endtask
  initial begin
    repeat(4)@(negedge aclk);aresetn=1;
    // One-beat transactions exercise consecutive first/last and accumulator bypass.
    for(int j=0;j<96;j++)send(1,j%19==0,0);
    for(int j=0;j<60;j++)send(j%25+1,j%17==0,1);
    send(216,0,0); // Ci=192, K=3, full reduction, dense signed values.
    wait(received==sent);repeat(6)@(negedge aclk);
    send(9,0,0);@(negedge aclk);aresetn=0;s_valid=0;
    repeat(4)@(negedge aclk);sent=0;received=0;aresetn=1;
    send(216,0,1);send(2,1,0);wait(received==sent);repeat(6)@(negedge aclk);
    $display("TB_PASS conv_array P=%0d pre_reset_transactions=157 recovery=%0d",P,received);$finish;
  end
  initial begin #4000000;$fatal(1,"array watchdog");end
endmodule

module tb_conv_array_p4;
  tb_conv_array #(.P(4)) test();
endmodule

module tb_conv_array_p6;
  tb_conv_array #(.P(6)) test();
endmodule
