`timescale 1ns/1ps
module tb_axis_bytes;
  logic aclk=0,aresetn=0;
  logic [7:0] s_data=0;
  logic s_valid=0,s_last=0,s_ready;
  logic [63:0] wire_data;
  logic [7:0] wire_keep;
  logic wire_valid,wire_last,wire_ready;
  logic [7:0] m_data;
  logic m_valid,m_last,m_ready=0,protocol_error;
  logic inject=0;
  logic [7:0] bad_keep=0;
  logic bad_last=0,bad_valid=0;
  logic force_block=0;
  logic [8:0] expected_q[$],value,held;
  logic held_valid=0;
  logic [72:0] held_axis;
  logic held_axis_valid=0;
  int bytes=0,packets=0;
  int unsigned random_state=32'h09262026;
  axis_byte_pack packer(.aclk,.aresetn,.s_data,.s_valid,.s_last,.s_ready,
    .m_axis_tdata(wire_data),.m_axis_tkeep(wire_keep),.m_axis_tvalid(wire_valid),
    .m_axis_tlast(wire_last),.m_axis_tready(wire_ready && !inject));
  axis_byte_unpack unpacker(.aclk,.aresetn,
    .s_axis_tdata(inject ? 64'hfefdfcfbfaf9f8f7 : wire_data),
    .s_axis_tkeep(inject ? bad_keep : wire_keep),.s_axis_tvalid(inject ? bad_valid : wire_valid),
    .s_axis_tlast(inject ? bad_last : wire_last),.s_axis_tready(wire_ready),
    .m_data,.m_valid,.m_last,.m_ready,.protocol_error);
  always #5 aclk=~aclk;
  function automatic int unsigned random_next(input int unsigned state);
    state^=state<<13; state^=state>>17; state^=state<<5; return state;
  endfunction
  always @(negedge aclk) begin
    random_state=random_next(random_state);
    m_ready=!force_block && (random_state[0] || random_state[1]);
  end
  always @(posedge aclk) begin
    if (aresetn) begin
      if (held_valid && (!m_valid || {m_last,m_data}!==held)) $fatal(1,"byte changed under backpressure");
      held_valid=m_valid && !m_ready; held={m_last,m_data};
      if (held_axis_valid && (!wire_valid || {wire_last,wire_keep,wire_data}!==held_axis))
        $fatal(1,"AXIS changed under backpressure");
      held_axis_valid=wire_valid && !wire_ready && !inject;
      held_axis={wire_last,wire_keep,wire_data};
      if (m_valid && m_ready) begin
        if (expected_q.size()==0) $fatal(1,"unexpected output byte");
        value=expected_q.pop_front();
        if ({m_last,m_data}!==value) $fatal(1,"byte mismatch actual=%h expected=%h",{m_last,m_data},value);
        bytes++; if (m_last) packets++;
      end
    end else begin held_valid=0; held_axis_valid=0; end
  end
  task automatic packet(input int length);
    for (int i=0;i<length;i++) begin
      @(negedge aclk);
      s_data=(i*37+length)%256; s_last=i==length-1; s_valid=1;
      expected_q.push_back({s_last,s_data});
      do @(posedge aclk); while (!s_ready);
      @(negedge aclk); s_valid=0;
      if (random_state[4:2]==0) repeat(3) @(negedge aclk);
    end
  endtask
  task automatic reset_all;
    @(negedge aclk); aresetn=0; s_valid=0; bad_valid=0;
    expected_q.delete(); repeat(3) @(negedge aclk); aresetn=1;
  endtask
  initial begin
    repeat(3) @(negedge aclk); aresetn=1;
    for (int n=1;n<=33;n++) packet(n);
    packet(1000); packet(65536);
    wait(expected_q.size()==0); repeat(4) @(negedge aclk);
    if (protocol_error || packets!=35) $fatal(1,"packet loss/protocol error");
    // Reset drops partial packer word and held unpacker transaction.
    s_data=8'hfa; s_valid=1; s_last=0;
    repeat(3) @(negedge aclk); s_valid=0;
    reset_all(); packet(7);
    wait(expected_q.size()==0); repeat(4) @(negedge aclk);
    force_block=1; packet(8); repeat(5) @(negedge aclk);
    if (!m_valid || expected_q.size()!=8) $fatal(1,"missing blocked reset transaction");
    reset_all(); force_block=0; packet(9);
    wait(expected_q.size()==0); repeat(4) @(negedge aclk);
    for (int n=0;n<3;n++) begin
      inject=1;
      bad_keep=n==0 ? 8'h00 : n==1 ? 8'h05 : 8'h07;
      bad_last=n!=2; bad_valid=1;
      @(negedge aclk); bad_valid=0;
      repeat(4) @(negedge aclk);
      if (!protocol_error || wire_ready || m_valid) $fatal(1,"malformed TKEEP not rejected");
      reset_all(); inject=0; packet(8);
      wait(expected_q.size()==0); repeat(4) @(negedge aclk);
    end
    if (packets!=40 || protocol_error) $fatal(1,"recovery failed");
    $display("TB_PASS tb_axis_bytes packets=%0d bytes=%0d malformed=3",packets,bytes); $finish;
  end
  initial begin #50000000; $fatal(1,"watchdog"); end
endmodule
