// Fresh Skin8 implementation: 2 spatial positions, 8 outputs, 8 inputs.
// Transactions are contiguous beats from first to last, without interleaving.
// Zero-fill inactive input lanes. Ignore inactive output lanes downstream.
// A tag identifies the result; it must remain constant within a transaction.
// All stages freeze on output backpressure. Reset invalidates in-flight work.
module dot128 #(parameter integer TAG_W = 16) (
  input logic clk, resetn,
  input logic [127:0] s_x,
  input logic [511:0] s_w,
  input logic [255:0] s_bias,
  input logic [TAG_W-1:0] s_tag,
  input logic s_first, s_last, s_valid,
  output logic s_ready,
  output logic [511:0] m_acc,
  output logic [TAG_W-1:0] m_tag,
  output logic m_overflow, m_valid,
  input logic m_ready
);
  (* use_dsp = "yes" *) logic signed [15:0] prod [0:15][0:7];
  (* use_dsp = "no" *) logic signed [16:0] pair_sum [0:15][0:3];
  (* use_dsp = "no" *) logic signed [17:0] quad_sum [0:15][0:1];
  (* use_dsp = "no" *) logic signed [18:0] oct_sum [0:15];
  (* use_dsp = "no" *) logic signed [31:0] accum [0:15];
  (* use_dsp = "no" *) logic signed [32:0] next_acc [0:15];
  logic [127:0] x_q;
  logic [511:0] w_q;
  logic [4:0] valid_q, first_q, last_q;
  logic [255:0] bias_q [0:4];
  logic [TAG_W-1:0] tag_q [0:4];
  logic advance, overflow_now, overflow_q;
  assign advance = !m_valid || m_ready;
  assign s_ready = resetn && advance;
  always_comb begin
    overflow_now = 0;
    for (int n=0; n<16; n++) begin
      next_acc[n] = (first_q[4] ?
        $signed({bias_q[4][(n%8)*32+31], bias_q[4][(n%8)*32+:32]}) :
        $signed({accum[n][31],accum[n]})) +
        $signed({{14{oct_sum[n][18]}},oct_sum[n]});
      overflow_now |= next_acc[n][32] != next_acc[n][31];
    end
  end
  always_ff @(posedge clk) begin
    if (!resetn) begin
      valid_q <= 0;
      m_valid <= 0;
      overflow_q <= 0;
    end else if (advance) begin
      valid_q <= {valid_q[3:0],s_valid};
      first_q <= {first_q[3:0],s_first};
      last_q <= {last_q[3:0],s_last};
      bias_q[0] <= s_bias;
      tag_q[0] <= s_tag;
      for (int st=1; st<5; st++) begin
        bias_q[st] <= bias_q[st-1];
        tag_q[st] <= tag_q[st-1];
      end
      m_valid <= valid_q[4] && last_q[4];
      if (s_valid) begin x_q<=s_x;w_q<=s_w;end
      if (valid_q[0])
        for (int p=0; p<2; p++)
          for (int o=0; o<8; o++)
            for (int i=0; i<8; i++)
              prod[p*8+o][i] <= $signed(x_q[(p*8+i)*8+:8]) * $signed(w_q[(o*8+i)*8+:8]);
      if (valid_q[1])
        for (int n=0; n<16; n++)
          for (int i=0; i<4; i++)
            pair_sum[n][i] <= $signed({prod[n][i*2][15],prod[n][i*2]}) +
                              $signed({prod[n][i*2+1][15],prod[n][i*2+1]});
      if (valid_q[2])
        for (int n=0; n<16; n++)
          for (int i=0; i<2; i++)
            quad_sum[n][i] <= $signed({pair_sum[n][i*2][16],pair_sum[n][i*2]}) +
                              $signed({pair_sum[n][i*2+1][16],pair_sum[n][i*2+1]});
      if (valid_q[3])
        for (int n=0; n<16; n++)
          oct_sum[n] <= $signed({quad_sum[n][0][17],quad_sum[n][0]}) +
                        $signed({quad_sum[n][1][17],quad_sum[n][1]});
      if (valid_q[4]) begin
        overflow_q <= (first_q[4] ? 1'b0 : overflow_q) | overflow_now;
        for (int n=0; n<16; n++) begin
          accum[n] <= next_acc[n][31:0];
          if (last_q[4]) m_acc[n*32+:32] <= next_acc[n][31:0];
        end
        if (last_q[4]) begin
          m_tag <= tag_q[4];
          m_overflow <= (first_q[4] ? 1'b0 : overflow_q) | overflow_now;
        end
      end
    end
  end
endmodule
