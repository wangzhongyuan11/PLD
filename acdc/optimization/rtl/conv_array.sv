// Spatially parallel exact dot products. P=4 => 256 logical INT8 multiplies,
// 128 DSP48E1; P=6 => 384 logical multiplies, 192 DSP48E1.
// Each accepted beat contributes eight Ci terms to eight Co at P positions.
// Bias enters once on first; only last publishes the complete INT32 sums.
module conv_array #(
  parameter integer P=4,
  parameter integer TAG_WIDTH=32
) (
  input logic aclk,aresetn,
  input logic [P*64-1:0] s_input,
  input logic [511:0] s_weights,
  input logic [255:0] s_bias,
  input logic [TAG_WIDTH-1:0] s_tag,
  input logic s_first,s_last,s_valid,
  output logic s_ready,
  output logic [P*256-1:0] m_acc,
  output logic [TAG_WIDTH-1:0] m_tag,
  output logic m_overflow,m_valid,
  input logic m_ready
);
  logic signed [24:0] packed_a [P/2][8];
  logic signed [17:0] weight_q [8][8];
  (* use_dsp="yes" *) logic signed [42:0] product [P/2][8][8];
  (* use_dsp="no" *) logic signed [15:0] split_product [P][8][8];
  (* use_dsp="no" *) logic signed [16:0] pair_sum [P][8][4];
  (* use_dsp="no" *) logic signed [17:0] quarter_sum [P][8][2];
  (* use_dsp="no" *) logic signed [18:0] dot_sum [P][8];
  (* use_dsp="no" *) logic signed [31:0] acc [P][8];
  (* use_dsp="no" *) logic signed [32:0] next_acc [P][8];
  logic [5:0] valid,first,last;
  logic [255:0] bias_pipe [6];
  logic [TAG_WIDTH-1:0] tag_pipe [6];
  logic advance,overflow_q,overflow_now;
  assign advance=!m_valid||m_ready;
  assign s_ready=advance;
  initial if(P!=4&&P!=6)$fatal(1,"conv_array P must be 4 or 6");
  always_comb begin
    overflow_now=0;
    for(int p=0;p<P;p++)for(int o=0;o<8;o++)begin
      next_acc[p][o]=(first[5] ? $signed({bias_pipe[5][o*32+31],bias_pipe[5][o*32+:32]}) :
                         $signed({acc[p][o][31],acc[p][o]}))
                         +$signed({{14{dot_sum[p][o][18]}},dot_sum[p][o]});
      overflow_now|=next_acc[p][o][32]!=next_acc[p][o][31];
    end
  end
  always_ff @(posedge aclk) begin
    if(!aresetn)begin valid<=0;m_valid<=0;overflow_q<=0;end
    else if(advance)begin
      valid<={valid[4:0],s_valid};
      first<={first[4:0],s_first};last<={last[4:0],s_last};
      m_valid<=valid[5]&&last[5];
      if(s_valid)begin
        bias_pipe[0]<=s_bias;
        tag_pipe[0]<=s_tag;
        for(int p=0;p<P/2;p++)for(int i=0;i<8;i++)begin
          packed_a[p][i]<=($signed({{17{s_input[((2*p+1)*8+i)*8+7]}},s_input[((2*p+1)*8+i)*8+:8]}) <<< 16)
                        +$signed({{17{s_input[(2*p*8+i)*8+7]}},s_input[(2*p*8+i)*8+:8]});
        end
        for(int o=0;o<8;o++)for(int i=0;i<8;i++)
          weight_q[o][i]<={{10{s_weights[(o*8+i)*8+7]}},s_weights[(o*8+i)*8+:8]};
      end
      for(int stage=1;stage<6;stage++)if(valid[stage-1])begin
        bias_pipe[stage]<=bias_pipe[stage-1];tag_pipe[stage]<=tag_pipe[stage-1];
      end
      if(valid[5]&&last[5])m_tag<=tag_pipe[5];
      if(valid[0])for(int p=0;p<P/2;p++)for(int o=0;o<8;o++)for(int i=0;i<8;i++)
        product[p][o][i]<=packed_a[p][i]*weight_q[o][i];
      if(valid[1])for(int p=0;p<P/2;p++)for(int o=0;o<8;o++)for(int i=0;i<8;i++)begin
        split_product[2*p][o][i]<=product[p][o][i][15:0];
        split_product[2*p+1][o][i]<=$signed(product[p][o][i][31:16])+$signed({15'd0,product[p][o][i][15]});
      end
      for(int p=0;p<P;p++)for(int o=0;o<8;o++)begin
        if(valid[2])for(int i=0;i<4;i++)
          pair_sum[p][o][i]<=$signed({split_product[p][o][2*i][15],split_product[p][o][2*i]})
                            +$signed({split_product[p][o][2*i+1][15],split_product[p][o][2*i+1]});
        if(valid[3])for(int i=0;i<2;i++)
          quarter_sum[p][o][i]<=$signed({pair_sum[p][o][2*i][16],pair_sum[p][o][2*i]})
                               +$signed({pair_sum[p][o][2*i+1][16],pair_sum[p][o][2*i+1]});
        if(valid[4])dot_sum[p][o]<=$signed({quarter_sum[p][o][0][17],quarter_sum[p][o][0]})
                                  +$signed({quarter_sum[p][o][1][17],quarter_sum[p][o][1]});
        if(valid[5])begin
          acc[p][o]<=next_acc[p][o][31:0];
          if(last[5])m_acc[(p*8+o)*32+:32]<=next_acc[p][o][31:0];
        end
      end
      if(valid[5])begin
        overflow_q<=(first[5]?1'b0:overflow_q)|overflow_now;
        if(last[5])m_overflow<=(first[5]?1'b0:overflow_q)|overflow_now;
      end
    end
  end
endmodule
