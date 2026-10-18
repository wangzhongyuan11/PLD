// Two exact signed INT8 products sharing a weight, using one DSP48E1.
// All stages freeze under backpressure. Decode products BEFORE accumulation.
module int8_pair_mul (
  input logic aclk, aresetn,
  input logic signed [7:0] s_a0, s_a1, s_weight,
  input logic s_valid,
  output logic s_ready,
  output logic signed [15:0] m_p0, m_p1,
  output logic m_valid,
  input logic m_ready
);
  logic signed [24:0] packed_a;
  logic signed [17:0] weight_q;
  (* use_dsp="yes" *) logic signed [42:0] product;
  logic [1:0] valid;
  logic advance;
  assign advance=!m_valid || m_ready;
  assign s_ready=advance;
  always_ff @(posedge aclk) begin
    if (!aresetn) begin valid<=0; m_valid<=0; end
    else if (advance) begin
      valid<={valid[0],s_valid};
      m_valid<=valid[1];
      if (s_valid) begin
        packed_a<=($signed({{17{s_a1[7]}},s_a1}) <<< 16)
                    +$signed({{17{s_a0[7]}},s_a0});
        weight_q<={{10{s_weight[7]}},s_weight};
      end
      if (valid[0]) product<=packed_a*weight_q;
      if (valid[1]) begin
        m_p0<=product[15:0];
        m_p1<=$signed(product[31:16])+$signed({15'd0,product[15]});
      end
    end
  end
endmodule
