// Four input-channel products per output channel per cycle.
// A dot-product transaction starts with s_first (loads bias) and ends with
// s_last (produces accumulators). Whole pipeline stalls under backpressure.
module conv_mac_array #(
  parameter int OUT_LANES = 8
) (
  input  logic                       aclk,
  input  logic                       aresetn,
  input  logic [31:0]                s_input,
  input  logic [OUT_LANES*32-1:0]    s_weights,
  input  logic [OUT_LANES*32-1:0]    s_bias,
  input  logic                       s_first,
  input  logic                       s_last,
  input  logic                       s_valid,
  output logic                       s_ready,
  output logic [OUT_LANES*32-1:0]    m_acc,
  output logic                       m_overflow,
  output logic                       m_valid,
  input  logic                       m_ready
);
  (* use_dsp = "yes" *) logic signed [15:0] products [OUT_LANES][4];
  logic signed [17:0] sums [OUT_LANES];
  logic signed [31:0] accumulators [OUT_LANES];
  logic signed [32:0] next_acc [OUT_LANES];
  logic [OUT_LANES*32-1:0] bias_p, bias_s;
  logic valid_p, valid_s, first_p, last_p, first_s, last_s;
  logic advance, overflow_q, overflow_now;

  assign advance = !m_valid || m_ready;
  assign s_ready = advance;
  always_comb begin
    overflow_now = 1'b0;
    for (int o=0; o<OUT_LANES; o++) begin
      if (first_s)
        next_acc[o] = $signed({bias_s[o*32+31],bias_s[o*32+:32]}) +
                      $signed({{15{sums[o][17]}},sums[o]});
      else
        next_acc[o] = $signed({accumulators[o][31],accumulators[o]}) +
                      $signed({{15{sums[o][17]}},sums[o]});
      overflow_now |= next_acc[o][32] != next_acc[o][31];
    end
  end

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      valid_p <= 0;
      valid_s <= 0;
      m_valid <= 0;
      overflow_q <= 0;
    end else if (advance) begin
      valid_p <= s_valid;
      valid_s <= valid_p;
      m_valid <= valid_s && last_s;
      if (valid_s) overflow_q <= (first_s ? 1'b0 : overflow_q) | overflow_now;
    end
  end

  // The valid pipeline, not a reset on every arithmetic register, flushes work.
  always_ff @(posedge aclk) begin
    if (aresetn && advance) begin
      if (s_valid) begin
        first_p <= s_first;
        last_p <= s_last;
        bias_p <= s_bias;
        for (int o=0; o<OUT_LANES; o++)
          for (int i=0; i<4; i++)
            products[o][i] <= $signed(s_input[i*8+:8]) * $signed(s_weights[o*32+i*8+:8]);
      end
      if (valid_p) begin
        first_s <= first_p;
        last_s <= last_p;
        bias_s <= bias_p;
        for (int o=0; o<OUT_LANES; o++)
          sums[o] <= $signed({{2{products[o][0][15]}},products[o][0]}) +
                     $signed({{2{products[o][1][15]}},products[o][1]}) +
                     $signed({{2{products[o][2][15]}},products[o][2]}) +
                     $signed({{2{products[o][3][15]}},products[o][3]});
      end
      if (valid_s) begin
        for (int o=0; o<OUT_LANES; o++) begin
          accumulators[o] <= next_acc[o][31:0];
          if (last_s) m_acc[o*32+:32] <= next_acc[o][31:0];
        end
        if (last_s) m_overflow <= (first_s ? 1'b0 : overflow_q) | overflow_now;
      end
    end
  end
endmodule
