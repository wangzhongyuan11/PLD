`timescale 1ns/1ps
module tb_conv_mac_128;
  logic aclk=0,aresetn=0;
  logic [63:0] s_input=0;
  logic [1023:0] s_weights=0;
  logic [511:0] s_bias=0,m_acc;
  logic s_first=0,s_last=0,s_valid=0,s_ready,m_overflow,m_valid,m_ready=0;
  logic [512:0] expected[$],item;
  longint signed accum [16];
  longint signed term;
  bit overflow;
  int sent=0,received=0;
  int unsigned rng=32'hadcd2516;
  conv_mac_128 dut(.*);
  always #5 aclk=~aclk;
  function automatic int unsigned random_next(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk) begin
    rng=random_next(rng);m_ready=aresetn&&(rng[0]||rng[3]);
  end
  always @(posedge aclk) begin
    if (aresetn&&m_valid&&m_ready) begin
      if (!expected.size()) $fatal(1,"unexpected result");
      item=expected.pop_front();
      if ({m_overflow,m_acc}!==item) $fatal(1,"128 MAC mismatch received=%0d overflow=%b/%b lane0=%h/%h",received,m_overflow,item[512],m_acc[31:0],item[31:0]);
      received++;
    end
  end
  task automatic send(input int length,input bit inject_overflow);
    logic [512:0] next_expected;
    overflow=0;
    for (int tap=0;tap<length;tap++) begin
      @(negedge aclk);
      s_valid=0;
      repeat(rng%3) @(negedge aclk);
      for (int i=0;i<8;i++) begin
        rng=random_next(rng);s_input[i*8+:8]=$signed(int'(rng%255)-127);
      end
      for (int o=0;o<16;o++) begin
        rng=random_next(rng);s_bias[o*32+:32]=inject_overflow ? 32'h7fffffff : int'(rng%65536)-32768;
        if (!tap) accum[o]=$signed(s_bias[o*32+:32]);
        term=0;
        for (int i=0;i<8;i++) begin
          rng=random_next(rng);s_weights[(o*8+i)*8+:8]=inject_overflow ? 8'd127 : int'(rng%255)-127;
          term+=$signed(s_input[i*8+:8])*$signed(s_weights[(o*8+i)*8+:8]);
        end
        accum[o]+=term;
        if (accum[o]>64'sd2147483647 || accum[o]<-64'sd2147483648) overflow=1;
        accum[o]=$signed(32'(accum[o]));
      end
      s_first=tap==0;s_last=tap==length-1;s_valid=1;
      do @(posedge aclk);while(!s_ready);
      if (s_last) begin
        next_expected[512]=overflow;
        for(int o=0;o<16;o++)next_expected[o*32+:32]=32'(accum[o]);
        expected.push_back(next_expected);sent++;
      end
    end
    @(negedge aclk);s_valid=0;
  endtask
  initial begin
    repeat(4)@(negedge aclk);aresetn=1;
    for (int j=0;j<40;j++)send(j%9+1,j%13==0);
    wait(received==sent);repeat(5)@(negedge aclk);
    aresetn=0;repeat(3)@(negedge aclk);aresetn=1;
    send(64,0);wait(received==sent);
    $display("TB_PASS conv_mac_128 transactions=%0d useful_MAC_per_issue=128",received);$finish;
  end
  initial begin #2000000;$fatal(1,"128 MAC watchdog");end
endmodule
