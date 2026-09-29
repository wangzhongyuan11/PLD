// Stage-0 toolchain smoke test for axis_reg_slice.
// Self-checking: pushes random beats through with source gaps and sink
// backpressure, compares every accepted payload. Prints TB_PASS/TB_FAIL.
`timescale 1ns/1ps
module tb_axis_reg_slice;

  localparam int  DATA_W = 8;
  localparam time CLK_P  = 10ns;

  logic              aclk = 0;
  logic              aresetn = 0;
  logic [DATA_W-1:0] s_tdata = '0;
  logic              s_tvalid = 0;
  logic              s_tready;
  logic [DATA_W-1:0] m_tdata;
  logic              m_tvalid;
  logic              m_tready = 0;

  int errors = 0;
  int beats  = 0;

  logic [DATA_W-1:0] exp_q[$];

  axis_reg_slice #(.DATA_W(DATA_W)) dut (
    .aclk(aclk), .aresetn(aresetn),
    .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
    .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready)
  );

  always #(CLK_P/2) aclk = ~aclk;

  // Monitor: sample every accepted beat and compare against the queue.
  initial forever begin
    @(posedge aclk);
    if (m_tvalid && m_tready) begin
      beats++;
      if (exp_q.size() == 0) begin
        $display("[%0t] ERROR: unexpected beat data=%02x", $time, m_tdata);
        errors++;
      end else begin
        automatic logic [DATA_W-1:0] exp = exp_q.pop_front();
        if (m_tdata !== exp) begin
          $display("[%0t] ERROR: beat %0d data=%02x exp=%02x", $time, beats, m_tdata, exp);
          errors++;
        end
      end
    end
  end

  // Sink: deterministic backpressure pattern (3 idle cycles, 1 ready).
  int ready_cnt = 0;
  initial forever begin
    @(negedge aclk);
    ready_cnt++;
    m_tready <= (ready_cnt % 4) == 0;
  end

  task automatic send_beat(input logic [DATA_W-1:0] d, input int vprob);
    s_tdata <= d;
    forever begin
      s_tvalid <= (vprob >= 100) || ($urandom_range(99) < vprob);
      @(posedge aclk);
      if (s_tvalid && s_tready) break;
    end
    s_tvalid <= 1'b0;
  endtask

  task automatic run_phase(input int n, input int vprob, input string name);
    for (int i = 0; i < n; i++) begin
      automatic logic [DATA_W-1:0] d = $urandom_range((1<<DATA_W)-1);
      exp_q.push_back(d);
      send_beat(d, vprob);
    end
    // drain: all queued beats must leave within a reasonable window
    repeat (4 * n + 64) @(posedge aclk);
    if (exp_q.size() != 0) begin
      $display("phase %s: FAIL, %0d beats stuck in DUT", name, exp_q.size());
      errors++;
    end else begin
      $display("phase %s: OK (%0d beats)", name, n);
    end
  endtask

  initial begin
    repeat (8) @(negedge aclk);
    aresetn = 1;

    run_phase(16,  100, "b2b");
    run_phase(128,  60, "src-gaps");
    run_phase(128, 100, "full-rate");

    repeat (16) @(posedge aclk);
    if (errors == 0 && beats == 16 + 128 + 128) begin
      $display("TB_PASS tb_axis_reg_slice beats=%0d", beats);
      $finish;
    end else begin
      $display("TB_FAIL tb_axis_reg_slice errors=%0d beats=%0d", errors, beats);
      $fatal(1, "self-check failed");
    end
  end

  // Watchdog
  initial begin
    #20ms;
    $display("TB_FAIL tb_axis_reg_slice timeout beats=%0d expq=%0d sv=%b sr=%b mv=%b mr=%b",
             beats, exp_q.size(), s_tvalid, s_tready, m_tvalid, m_tready);
    $fatal(1, "watchdog");
  end

endmodule
