// Full signed INT32 x INT32 product, Q24 ties-even, saturation [-127,127].
// Five globally stalled pipeline stages, one transaction per clock at full rate.
// Multipliers in the model are nonnegative; arithmetic also handles signed bits.
module requant_q24 (
  input  logic               aclk,
  input  logic               aresetn,
  input  logic signed [31:0] s_acc,
  input  logic signed [31:0] s_multiplier,
  input  logic               s_valid,
  output logic               s_ready,
  output logic signed [7:0]  m_value,
  output logic               m_valid,
  input  logic               m_ready
);
  (* use_dsp = "yes" *) logic signed [33:0] ll, lh, hl, hh;
  logic signed [63:0] low_sum, high_sum, product_q;
  logic valid_input, valid_p, valid_s, valid_product;
  logic signed [31:0] acc_q, multiplier_q;
  logic advance, increment;
  logic signed [39:0] floor_value;
  logic signed [7:0] quantized;

  assign advance = !m_valid || m_ready;
  assign s_ready = advance;
  always_comb begin
    floor_value = $signed(product_q[63:24]);
    increment = (product_q[23:0] > 24'h800000) ||
                ((product_q[23:0] == 24'h800000) && product_q[24]);
    // Saturate by the floor first; rounding only needs an 8-bit addition.
    // floor=-128 always saturates to -127, whether it rounds up or not.
    if (floor_value >= 40'sd127) quantized = 8'sd127;
    else if (floor_value < -40'sd127) quantized = -8'sd127;
    else quantized = product_q[31:24] + {7'd0,increment};
  end

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      valid_input <= 0;
      valid_p <= 0;
      valid_s <= 0;
      valid_product <= 0;
      m_valid <= 0;
    end else if (advance) begin
      valid_input <= s_valid;
      valid_p <= valid_input;
      valid_s <= valid_p;
      valid_product <= valid_s;
      m_valid <= valid_product;
    end
  end

  // Split at bit 16. Low halves are unsigned, high halves signed.
  // Register each partial product so no DSP-to-DSP combinational cascade
  // implements a whole 32x32 multiply inside a single 100 MHz cycle.
  always_ff @(posedge aclk) begin
    if (aresetn && advance) begin
      if (s_valid) begin
        acc_q <= s_acc; multiplier_q <= s_multiplier;
      end
      if (valid_input) begin
        ll <= $signed({1'b0,acc_q[15:0]}) * $signed({1'b0,multiplier_q[15:0]});
        lh <= $signed({1'b0,acc_q[15:0]}) * $signed({multiplier_q[31],multiplier_q[31:16]});
        hl <= $signed({acc_q[31],acc_q[31:16]}) * $signed({1'b0,multiplier_q[15:0]});
        hh <= $signed({acc_q[31],acc_q[31:16]}) * $signed({multiplier_q[31],multiplier_q[31:16]});
      end
      if (valid_p) begin
        low_sum <= $signed({{30{ll[33]}},ll}) + ($signed({{30{lh[33]}},lh}) <<< 16);
        high_sum <= $signed({{30{hl[33]}},hl}) + ($signed({{30{hh[33]}},hh}) <<< 16);
      end
      if (valid_s) product_q <= low_sum + (high_sum <<< 16);
      if (valid_product) m_value <= quantized;
    end
  end
endmodule
