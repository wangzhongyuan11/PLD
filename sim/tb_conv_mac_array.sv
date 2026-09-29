`timescale 1ns/1ps
module tb_conv_mac_array;
  logic aclk=0, aresetn=0;
  logic [31:0] s_input=0;
  logic [255:0] s_weights=0, s_bias=0, m_acc;
  logic s_first=0, s_last=0, s_valid=0, s_ready;
  logic m_overflow, m_valid, m_ready=0;
  logic [256:0] expected_q[$];
  logic [256:0] expected_value, held_value;
  logic held=0;
  int fd, rc, beats=0, packets=0;
  int unsigned random_state=32'habcdef01;
  logic first_read, last_read, overflow_read;
  logic [31:0] input_read;
  logic [255:0] weights_read, bias_read, acc_read;
  conv_mac_array dut(.*);
  always #5 aclk=~aclk;

  function automatic int unsigned next_random(input int unsigned state);
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
  endfunction

  always @(posedge aclk) begin
    if (aresetn) begin
      if (held && (!m_valid || {m_overflow,m_acc} !== held_value))
        $fatal(1,"MAC changed while blocked");
      held=m_valid && !m_ready;
      held_value={m_overflow,m_acc};
      if (m_valid && m_ready) begin
        if (expected_q.size()==0) $fatal(1,"unexpected MAC output");
        expected_value=expected_q.pop_front();
        if ({m_overflow,m_acc} !== expected_value)
          $fatal(1,"MAC mismatch packet=%0d actual=%h expected=%h",packets,{m_overflow,m_acc},expected_value);
        packets++;
      end
    end else held=0;
  end

  initial begin
    repeat(4) @(negedge aclk);
    aresetn=1;
    fd=$fopen("../vectors/mac.txt","r");
    if (!fd) $fatal(1,"missing mac vectors");
    while (!$feof(fd)) begin
      rc=$fscanf(fd,"%h %h %h %h %h %h %h\n",first_read,last_read,input_read,weights_read,bias_read,acc_read,overflow_read);
      if (rc==7) begin
        @(negedge aclk);
        s_first=first_read; s_last=last_read; s_input=input_read;
        s_weights=weights_read; s_bias=bias_read; s_valid=1;
        if (last_read) expected_q.push_back({overflow_read,acc_read});
        do begin
          random_state=next_random(random_state);
          m_ready=random_state[0] || random_state[1];
          @(posedge aclk);
          if (!s_ready) @(negedge aclk);
        end while (!s_ready);
        @(negedge aclk);
        s_valid=0;
        beats++;
        if (random_state[3:2]==0) repeat(2) @(negedge aclk);
      end else if (!$feof(fd)) $fatal(1,"malformed MAC vector");
    end
    $fclose(fd);
    m_ready=1;
    wait(expected_q.size()==0);
    repeat(5) @(negedge aclk);
    if (packets!=132) $fatal(1,"lost packet");
    // Occupy the pipeline/output while blocked, then discard all on reset.
    m_ready=0; s_valid=1; s_first=1; s_last=1;
    repeat(6) @(negedge aclk);
    aresetn=0; s_valid=0;
    repeat(3) @(negedge aclk);
    aresetn=1; m_ready=1;
    repeat(5) @(negedge aclk);
    // Zero dot product after reset proves accumulator reinitialization.
    s_input=0; s_weights=0; s_bias=0; s_first=1; s_last=1; s_valid=1;
    expected_q.push_back(257'd0);
    @(negedge aclk); s_valid=0;
    wait(expected_q.size()==0);
    repeat(3) @(negedge aclk);
    // Full-rate dot product: 32 consecutive beats, four products per lane.
    for (int i=0; i<32; i++) begin
      s_valid=1; s_first=(i==0); s_last=(i==31);
      s_input=32'h01010101; s_weights={32{8'h01}}; s_bias=0;
      if (i==31) expected_q.push_back({1'b0,{8{32'd128}}});
      @(negedge aclk);
    end
    // Consecutive single-beat dots exercise independent pipeline transactions.
    for (int i=0; i<32; i++) begin
      s_first=1; s_last=1; s_bias={8{32'd5}};
      expected_q.push_back({1'b0,{8{32'd9}}});
      @(negedge aclk);
    end
    s_valid=0;
    wait(expected_q.size()==0);
    repeat(3) @(negedge aclk);
    if (packets!=166) $fatal(1,"full-rate MAC packets lost");
    $display("TB_PASS tb_conv_mac_array beats=%0d packets=%0d overflow=PASS reset=PASS",beats,packets);
    $finish;
  end
  initial begin #20000000; $fatal(1,"MAC watchdog"); end
endmodule
