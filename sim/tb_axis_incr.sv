// Full-coverage self-checking TB for axis_incr (stage-2 placeholder IP).
//
// Coverage:
//   - data: all-0xFF (wrap), all-0x00, 0x7F boundary, random
//   - tkeep: all-ones, byte-lane tail beats (DMA-style non-multiple lengths),
//     random lanes; must pass through bit-exact
//   - packets: 1-beat, 1..4-beat back-to-back, 1024-beat single stream
//   - handshake: full-rate b2b, source gaps, sink backpressure, both random
//   - reset injected mid-drain: internal state must flush, no stale beat
//   - watchdog on progress loss
// Reference model is re-implemented here (byte+1 mod 256), not shared with DUT.
`timescale 1ns/1ps
module tb_axis_incr;

  localparam int  DATA_W = 64;
  localparam int  BYTES  = DATA_W/8;
  localparam int  KEEP_W = BYTES;
  localparam time CLK_P  = 10ns;

  logic                  aclk = 0;
  logic                  aresetn = 0;
  logic [DATA_W-1:0]     s_tdata = '0;
  logic [KEEP_W-1:0]     s_tkeep = '1;
  logic                  s_tvalid = 0;
  logic                  s_tlast = 0;
  logic                  s_tready;
  logic [DATA_W-1:0]     m_tdata;
  logic [KEEP_W-1:0]     m_tkeep;
  logic                  m_tvalid;
  logic                  m_tlast;
  logic                  m_tready = 0;

  int errors = 0;
  int beats  = 0;
  int pkts   = 0;
  int last_beats = -1;

  int sink_prob = 100;  // % chance tready is high each cycle

  logic [DATA_W-1:0] exp_data_q[$];
  logic [KEEP_W-1:0] exp_keep_q[$];
  logic              exp_last_q[$];

  axis_incr #(.DATA_W(DATA_W)) dut (
    .aclk(aclk), .aresetn(aresetn),
    .s_axis_tdata(s_tdata), .s_axis_tkeep(s_tkeep),
    .s_axis_tvalid(s_tvalid), .s_axis_tlast(s_tlast), .s_axis_tready(s_tready),
    .m_axis_tdata(m_tdata), .m_axis_tkeep(m_tkeep),
    .m_axis_tvalid(m_tvalid), .m_axis_tlast(m_tlast), .m_axis_tready(m_tready)
  );

  always #(CLK_P/2) aclk = ~aclk;

  function automatic logic [DATA_W-1:0] ref_incr(input logic [DATA_W-1:0] d);
    ref_incr = '0;
    for (int b = 0; b < BYTES; b++)
      ref_incr[8*b +: 8] = (d[8*b +: 8] + 8'd1) & 8'hFF;
  endfunction

  // Monitor: compare every accepted beat against the expected queues.
  task automatic mon_beat();
    logic [DATA_W-1:0] e_data;
    logic [KEEP_W-1:0] e_keep;
    logic              e_last;
    beats++;
    if (m_tlast) pkts++;
    if (exp_data_q.size() == 0) begin
      $display("[%0t] ERROR: unexpected beat data=%0h keep=%0h last=%0b",
               $time, m_tdata, m_tkeep, m_tlast);
      errors++;
    end else begin
      e_data = exp_data_q.pop_front();
      e_keep = exp_keep_q.pop_front();
      e_last = exp_last_q.pop_front();
      if (m_tdata !== e_data || m_tkeep !== e_keep || m_tlast !== e_last) begin
        $display("[%0t] ERROR: beat %0d got(d=%0h k=%0h l=%0b) exp(d=%0h k=%0h l=%0b)",
                 $time, beats, m_tdata, m_tkeep, m_tlast, e_data, e_keep, e_last);
        errors++;
      end
    end
  endtask

  initial forever begin
    @(posedge aclk);
    if (m_tvalid && m_tready) mon_beat();
  end

  // Sink: probabilistic backpressure, pattern set per phase.
  initial forever begin
    @(negedge aclk);
    m_tready <= (sink_prob >= 100) || ($urandom_range(99) < sink_prob);
  end

  // Present one beat, hold until accepted. vprob = % cycles valid is high.
  task automatic send_beat(input logic [DATA_W-1:0] d, input logic [KEEP_W-1:0] k,
                           input logic last, input int vprob);
    s_tdata <= d;
    s_tkeep <= k;
    s_tlast <= last;
    forever begin
      s_tvalid <= (vprob >= 100) || ($urandom_range(99) < vprob);
      @(posedge aclk);
      if (s_tvalid && s_tready) break;
    end
    s_tvalid <= 1'b0;
  endtask

  // dmode 0=random 1=all-FF 2=all-00 3=all-7F 4=random with DMA-style tail keep
  // 5=random with fully random keep lanes
  function automatic logic [DATA_W-1:0] gen_data(input int mode);
    case (mode)
      1: gen_data = '1;
      2: gen_data = '0;
      3: gen_data = {BYTES{8'h7F}};
      default: gen_data = $urandom();
    endcase
  endfunction

  function automatic logic [KEEP_W-1:0] gen_keep(input int mode, input int i, input int len);
    logic [KEEP_W-1:0] k;
    case (mode)
      4: k = (i == len-1) ? {{(KEEP_W-1){1'b0}}, 1'b1} : {KEEP_W{1'b1}};
      5: k = $urandom();
      default: k = {KEEP_W{1'b1}};
    endcase
    gen_keep = k;
  endfunction

  task automatic send_packet(input int len, input int vprob, input int dmode);
    logic [DATA_W-1:0] d;
    logic [KEEP_W-1:0] k;
    for (int i = 0; i < len; i++) begin
      d = gen_data(dmode);
      k = gen_keep(dmode, i, len);
      exp_data_q.push_back(ref_incr(d));
      exp_keep_q.push_back(k);
      exp_last_q.push_back(i == len-1);
      send_beat(d, k, (i == len-1), vprob);
    end
  endtask

  // Wait until the expected queue drains (or stall-detected). ph = phase id.
  task automatic drain_check(input int ph);
    int idle;
    idle = 0;
    while (exp_data_q.size() > 0 && idle < 2000) begin
      @(posedge aclk);
      idle = (beats == last_beats) ? idle + 1 : 0;
      last_beats = beats;
    end
    if (exp_data_q.size() != 0) begin
      $display("phase %0d FAIL: %0d beats stuck", ph, exp_data_q.size());
      errors++;
    end else begin
      $display("phase %0d OK (beats=%0d pkts=%0d)", ph, beats, pkts);
    end
  endtask

  task automatic run_pkts(input int ph, input int npkts, input int maxlen,
                          input int vprob, input int dmode, input int sink_p);
    sink_prob = sink_p;
    for (int p = 0; p < npkts; p++)
      send_packet($urandom_range(maxlen-1) + 1, vprob, dmode);
    drain_check(ph);
  endtask

  initial begin : main
    repeat (8) @(negedge aclk);
    aresetn = 1;

    // directed: data corners
    send_packet(1, 100, 1);   drain_check(1);  // 1-beat all-FF (wrap)
    send_packet(1, 100, 2);   drain_check(2);  // 1-beat all-00
    send_packet(3, 100, 3);   drain_check(3);  // 3-beat all-7F
    // tkeep: DMA-style tail beats and random lanes
    run_pkts(4, 8, 7, 100, 4, 100);            // odd-length tails, keep=0x01
    run_pkts(5, 8, 9, 100, 5, 100);            // random keep lanes
    send_packet(1, 100, 4);                    // single partial beat
    drain_check(6);
    // packet shapes / handshake stress
    run_pkts(7, 20,   4, 100, 0, 100);         // short b2b
    send_packet(1024, 100, 0);                 // long full-rate
    drain_check(8);
    run_pkts(9, 8,   64, 100, 0,  25);         // backpressure
    run_pkts(10, 8,  64,  40, 0, 100);         // source gaps
    run_pkts(11, 6, 128, 100, 4, 100);         // tail-keep under full rate
    for (int s = 0; s < 8; s++)
      run_pkts(12+s, 6, 128, 50, 0, 50);       // both random

    // mid-drain reset: long packet under heavy backpressure, reset while
    // beats are still in flight; only the fresh packet sent after reset
    // may appear at the output.
    sink_prob = 10;
    send_packet(256, 100, 0);
    repeat (30) @(posedge aclk);       // partially drained, DUT non-empty
    aresetn = 0;
    repeat (5) @(negedge aclk);
    exp_data_q.delete();               // in-flight beats are dropped
    exp_keep_q.delete();
    exp_last_q.delete();
    if (m_tvalid !== 1'b0) begin
      $display("phase 20 FAIL: m_tvalid not cleared by reset");
      errors++;
    end
    aresetn = 1;
    repeat (4) @(negedge aclk);
    run_pkts(21, 4, 32, 100, 4, 100);  // post-reset integrity (tail keeps)

    repeat (64) @(posedge aclk);
    if (errors == 0) begin
      $display("TB_PASS tb_axis_incr beats=%0d pkts=%0d", beats, pkts);
      $finish;
    end else begin
      $display("TB_FAIL tb_axis_incr errors=%0d beats=%0d pkts=%0d", errors, beats, pkts);
      $fatal(1, "self-check failed");
    end
  end

  // Watchdog: no accepted beat for a long wall-clock window is fatal.
  initial begin
    #50ms;
    $display("TB_FAIL tb_axis_incr watchdog (beats=%0d)", beats);
    $fatal(1, "watchdog");
  end

endmodule
