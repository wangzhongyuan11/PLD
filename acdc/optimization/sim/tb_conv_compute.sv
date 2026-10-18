`timescale 1ns/1ps
module tb_conv_compute #(parameter integer P=4);
  logic aclk=0,aresetn=0;
  logic [P*64-1:0] s_input=0;
  logic [511:0] s_weights=0;
  logic [255:0] s_bias=0,s_multiplier=0;
  logic [P*8-1:0] s_lane_mask=0;
  logic [31:0] s_tag=0,m_data,m_tag;
  logic s_relu=0,s_first=0,s_last=0,s_valid=0,s_ready;
  logic [3:0] m_keep;
  logic m_first,m_last,m_overflow,m_valid,m_ready=0;
  logic [2:0] queued_groups;
  logic [70:0] expected[2048],held;
  bit stalled=0;
  bit force_pause=0;
  int input_fd,expected_fd,input_count,expected_count,rc,received=0;
  int blocked_cycles=0,full_cycles=0,cycles=0;
  int unsigned rng=32'h2564acdc;
  conv_compute #(.P(P),.DEPTH(4)) dut(.*);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk)begin
    rng=next_rng(rng);
    // Repeated long stalls must fill group FIFO and reach the input handshake.
    m_ready=aresetn&&!force_pause&&(cycles%900>500)&&(rng[0]||rng[6]);
  end
  always @(posedge aclk)begin
    if(!aresetn)begin stalled=0;received=0;end
    else begin
      cycles++;
      if(s_valid&&!s_ready)blocked_cycles++;
      if(queued_groups==4)full_cycles++;
      if(stalled&&(!m_valid||{m_tag,m_last,m_first,m_overflow,m_keep,m_data}!==held))
        $fatal(1,"compute result changed while stalled");
      stalled=m_valid&&!m_ready;held={m_tag,m_last,m_first,m_overflow,m_keep,m_data};
      if(m_valid&&m_ready)begin
        if(received>=expected_count)$fatal(1,"extra compute output");
        if({m_tag,m_last,m_first,m_overflow,m_keep,m_data}!==expected[received])
          $fatal(1,"compute mismatch P=%0d beat=%0d actual=%h expected=%h",P,received,
                 {m_tag,m_last,m_first,m_overflow,m_keep,m_data},expected[received]);
        received++;
      end
      if(dut.q_ready!=={4{dut.q_ready[0]}}||dut.q_valid!=={4{dut.q_valid[0]}})
        $fatal(1,"Q24 lanes lost lockstep");
    end
  end
  task automatic feed(input int count);
    for(int j=0;j<count;j++)begin
      @(negedge aclk);s_valid=0;
      if(j%13==0)repeat(j%3+1)@(negedge aclk);
      rc=$fscanf(input_fd,"%h %h %h %h %h %h %h %h %h\n",s_first,s_last,s_relu,s_tag,s_lane_mask,s_bias,s_multiplier,s_input,s_weights);
      if(rc!=9)$fatal(1,"short input vector %0d",j);
      s_valid=1;
      do @(posedge aclk);while(!s_ready);
    end
    @(negedge aclk);s_valid=0;
  endtask
  initial begin
    input_fd=$fopen("input.txt","r");expected_fd=$fopen("expected.txt","r");
    if(!input_fd||!expected_fd)$fatal(1,"missing compute vectors");
    rc=$fscanf(input_fd,"%d\n",input_count);if(rc!=1)$fatal(1,"bad input header");
    rc=$fscanf(expected_fd,"%d\n",expected_count);if(rc!=1||expected_count>2048)$fatal(1,"bad output header");
    for(int j=0;j<expected_count;j++)begin rc=$fscanf(expected_fd,"%h\n",expected[j]);if(rc!=1)$fatal(1,"short output vector");end
    repeat(4)@(negedge aclk);aresetn=1;
    feed(input_count);
    wait(received==expected_count);repeat(20)@(negedge aclk);
    $fclose(input_fd);input_fd=$fopen("input.txt","r");rc=$fscanf(input_fd,"%d\n",input_count);
    // Abort with array/FIFO/Q24 state in flight and output deliberately blocked.
    force_pause=1;feed(12);repeat(6)@(negedge aclk);
    aresetn=0;repeat(4)@(negedge aclk);aresetn=1;force_pause=0;
    @(negedge aclk);if(m_valid||queued_groups!=0)$fatal(1,"stale compute output after reset");
    $fclose(input_fd);input_fd=$fopen("input.txt","r");rc=$fscanf(input_fd,"%d\n",input_count);
    feed(input_count);
    wait(received==expected_count);repeat(20)@(negedge aclk);
    if(blocked_cycles==0||full_cycles==0)$fatal(1,"FIFO pressure coverage missing");
    $display("TB_PASS conv_compute P=%0d input_beats_per_pass=%0d output_beats_per_pass=%0d passes=2 reset_inflight=1 cycles=%0d blocked=%0d full=%0d",P,input_count,received,cycles,blocked_cycles,full_cycles);
    $finish;
  end
  initial begin #4000000;$fatal(1,"compute watchdog");end
endmodule

module tb_conv_compute_p4;
  tb_conv_compute #(.P(4)) test();
endmodule
module tb_conv_compute_p6;
  tb_conv_compute #(.P(6)) test();
endmodule
