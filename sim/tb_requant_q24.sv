`timescale 1ns/1ps
module tb_requant_q24;
  logic aclk=0, aresetn=0;
  logic signed [31:0] s_acc=0, s_multiplier=0;
  logic s_valid=0, s_ready, m_valid, m_ready=0;
  logic signed [7:0] m_value;
  logic [7:0] expected_q[$];
  logic [7:0] expected_value;
  logic [7:0] held_value;
  logic held=0;
  int fd, rc, count=0, received=0;
  int unsigned random_state=32'h31415926;
  logic [31:0] acc_read, mult_read;
  logic [7:0] value_read;
  string vector_path;

  requant_q24 dut(.*);
  always #5 aclk=~aclk;

  function automatic int unsigned next_random(input int unsigned state);
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  endfunction

  always @(posedge aclk) begin
    if (aresetn) begin
      if (held && (!m_valid || m_value !== held_value)) $fatal(1,"output changed during backpressure");
      held = m_valid && !m_ready;
      held_value = m_value;
      if (m_valid && m_ready) begin
        if (expected_q.size()==0) $fatal(1,"unexpected output");
        expected_value=expected_q.pop_front();
        if (m_value !== expected_value)
          $fatal(1,"requant mismatch item=%0d got=%0d expected=%0d",received,$signed(m_value),$signed(expected_value));
        received++;
      end
    end else held=0;
  end

  task automatic send(input logic [31:0] a, m, input logic [7:0] want);
    @(negedge aclk);
    s_acc=a; s_multiplier=m; s_valid=1;
    expected_q.push_back(want);
    do begin
      random_state=next_random(random_state);
      m_ready=random_state[0] || random_state[1];
      @(posedge aclk);
      if (!s_ready) @(negedge aclk);
    end while (!s_ready);
    @(negedge aclk);
    s_valid=0;
    count++;
    random_state=next_random(random_state);
    if (random_state[3:2]==0) repeat(2) @(negedge aclk);
  endtask

  initial begin
    repeat(4) @(negedge aclk);
    aresetn=1;
    if (!$value$plusargs("VECTORS=%s",vector_path)) vector_path="../vectors/requant.txt";
    fd=$fopen(vector_path,"r");
    if (!fd) $fatal(1,"cannot open %s",vector_path);
    while (!$feof(fd)) begin
      rc=$fscanf(fd,"%h %h %h\n",acc_read,mult_read,value_read);
      if (rc==3) send(acc_read,mult_read,value_read);
      else if (!$feof(fd)) $fatal(1,"malformed vector");
    end
    $fclose(fd);
    m_ready=1;
    wait(expected_q.size()==0);
    repeat(4) @(negedge aclk);
    if (count!=received) $fatal(1,"lost values");

    // Reset with both elastic stages occupied; stale results must disappear.
    m_ready=0; s_valid=1; s_acc=32'sd99; s_multiplier=32'sd16777216;
    repeat(4) @(negedge aclk);
    aresetn=0; s_valid=0;
    repeat(3) @(negedge aclk);
    aresetn=1; m_ready=1;
    repeat(4) @(negedge aclk);
    send(-32'sd3,32'sd8388608,8'hfe);
    m_ready=1;
    wait(expected_q.size()==0);
    repeat(3) @(negedge aclk);
    // Continuous one-beat-per-cycle throughput, alternating signed ties.
    for (int i=0; i<64; i++) begin
      s_valid=1;
      s_acc=(i & 1) ? -32'sd3 : 32'sd3;
      s_multiplier=32'sd8388608;
      expected_q.push_back((i & 1) ? 8'hfe : 8'h02);
      count++;
      @(negedge aclk);
    end
    s_valid=0;
    wait(expected_q.size()==0);
    repeat(3) @(negedge aclk);
    if (count!=received) $fatal(1,"full-rate values lost");
    $display("TB_PASS tb_requant_q24 vectors=%0d received=%0d reset_flush=PASS",count,received);
    $finish;
  end

  initial begin
    #20000000;
    $fatal(1,"requant watchdog");
  end
endmodule
