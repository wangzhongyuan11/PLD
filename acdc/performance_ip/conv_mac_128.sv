// Eight signed INT8 inputs x sixteen outputs. Entire arithmetic pipeline stalls
// under output backpressure. A transaction accumulates across ALL Ci/K taps;
// bias is included on s_first, quantization belongs downstream after s_last.
module conv_mac_128 (
  input logic aclk, aresetn,
  input logic [63:0] s_input,
  input logic [1023:0] s_weights,
  input logic [511:0] s_bias,
  input logic s_first,s_last,s_valid,
  output logic s_ready,
  output logic [511:0] m_acc,
  output logic m_overflow,m_valid,
  input logic m_ready
);
  (* use_dsp="yes" *) logic signed [15:0] product [16][8];
  (* use_dsp="no" *) logic signed [16:0] pair_sum [16][4];
  (* use_dsp="no" *) logic signed [17:0] quarter_sum [16][2];
  (* use_dsp="no" *) logic signed [18:0] dot_sum [16];
  (* use_dsp="no" *) logic signed [31:0] acc [16];
  (* use_dsp="no" *) logic signed [32:0] next_acc [16];
  logic [3:0] valid,first,last;
  logic [511:0] bias_pipe [4];
  logic overflow_q,overflow_now,advance;
  assign advance=!m_valid || m_ready;
  assign s_ready=advance;
  always_comb begin
    overflow_now=0;
    for (int o=0;o<16;o++) begin
      next_acc[o]=(first[3] ? $signed({bias_pipe[3][o*32+31],bias_pipe[3][o*32+:32]}) :
                                      $signed({acc[o][31],acc[o]})) +
                    $signed({{14{dot_sum[o][18]}},dot_sum[o]});
      overflow_now|=next_acc[o][32]!=next_acc[o][31];
    end
  end
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      valid<=0;m_valid<=0;overflow_q<=0;
    end else if (advance) begin
      valid<={valid[2:0],s_valid};
      first<={first[2:0],s_first};last<={last[2:0],s_last};
      m_valid<=valid[3]&&last[3];
      if (s_valid) begin
        bias_pipe[0]<=s_bias;
        for (int o=0;o<16;o++)
          for (int i=0;i<8;i++)
            product[o][i]<=$signed(s_input[i*8+:8])*$signed(s_weights[(o*8+i)*8+:8]);
      end
      for (int stage=1;stage<4;stage++) if (valid[stage-1]) bias_pipe[stage]<=bias_pipe[stage-1];
      for (int o=0;o<16;o++) begin
        if (valid[0]) for (int i=0;i<4;i++)
          pair_sum[o][i]<=$signed({product[o][2*i][15],product[o][2*i]})+
                          $signed({product[o][2*i+1][15],product[o][2*i+1]});
        if (valid[1]) for (int i=0;i<2;i++)
          quarter_sum[o][i]<=$signed({pair_sum[o][2*i][16],pair_sum[o][2*i]})+
                             $signed({pair_sum[o][2*i+1][16],pair_sum[o][2*i+1]});
        if (valid[2]) dot_sum[o]<=$signed({quarter_sum[o][0][17],quarter_sum[o][0]})+
                                $signed({quarter_sum[o][1][17],quarter_sum[o][1]});
        if (valid[3]) begin
          acc[o]<=next_acc[o][31:0];
          if (last[3]) m_acc[o*32+:32]<=next_acc[o][31:0];
        end
      end
      if (valid[3]) begin
        overflow_q<=(first[3]?1'b0:overflow_q)|overflow_now;
        if (last[3]) m_overflow<=(first[3]?1'b0:overflow_q)|overflow_now;
      end
    end
  end
endmodule
