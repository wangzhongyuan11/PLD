// Streaming exact convolution reduction + bounded group FIFO + four Q24 lanes.
// Array can process the next reduction while an earlier group is requantized.
// Result byte order: spatial position, then its eight output channels.
module conv_compute #(
  parameter integer P=4,
  parameter integer DEPTH=4
) (
  input logic aclk,aresetn,
  input logic [P*64-1:0] s_input,
  input logic [511:0] s_weights,
  input logic [255:0] s_bias,s_multiplier,
  input logic [P*8-1:0] s_lane_mask,
  input logic [31:0] s_tag,
  input logic s_relu,s_first,s_last,s_valid,
  output logic s_ready,
  output logic [31:0] m_data,m_tag,
  output logic [3:0] m_keep,
  output logic m_first,m_last,m_overflow,m_valid,
  input logic m_ready,
  output logic [$clog2(DEPTH+1)-1:0] queued_groups
);
  localparam integer LANES=P*8;
  localparam integer META=289+LANES;
  localparam integer PTR=$clog2(DEPTH);
  logic [P*256-1:0] array_acc,acc_fifo[DEPTH];
  logic [META-1:0] array_meta,meta_fifo[DEPTH];
  logic overflow_fifo[DEPTH];
  logic array_valid,array_ready,array_overflow,push,pop;
  logic [PTR-1:0] rp,wp;
  logic [$clog2(LANES)-1:0] index;
  logic [3:0] q_ready,q_valid;
  logic [31:0] quant_data;
  logic quant_accept,quant_advance;
  logic [38:0] token_pipe[5],token;
  logic [4:0] token_valid;
  logic [255:0] multipliers;
  logic [LANES-1:0] lane_mask;
  logic relu;
  logic [4:0] relu_pipe;
  assign array_ready=queued_groups<DEPTH;
  assign push=array_valid&&array_ready;
  assign multipliers=meta_fifo[rp][255:0];
  assign lane_mask=meta_fifo[rp][256+:LANES];
  assign relu=meta_fifo[rp][256+LANES];
  assign quant_accept=queued_groups!=0&&(&q_ready);
  assign pop=quant_accept&&index==LANES-4;
  assign quant_advance=!m_valid||m_ready;
  assign m_valid=&q_valid;
  assign m_data=quant_data;
  assign {m_tag,m_last,m_first,m_overflow,m_keep}=token_pipe[4];
  assign token={meta_fifo[rp][257+LANES+:32],index==LANES-4,index==0,
                overflow_fifo[rp],lane_mask[index+:4]};
  conv_array #(.P(P),.TAG_WIDTH(META)) array_core(
    .aclk,.aresetn,.s_input,.s_weights,.s_bias,
    .s_tag({s_tag,s_relu,s_lane_mask,s_multiplier}),.s_first,.s_last,.s_valid,.s_ready,
    .m_acc(array_acc),.m_tag(array_meta),.m_overflow(array_overflow),
    .m_valid(array_valid),.m_ready(array_ready));
  for(genvar q=0;q<4;q++)begin:quantizers
    logic signed [7:0] value;
    logic [31:0] multiplier;
    assign multiplier=multipliers[(((index+q)%8)*32)+:32];
    requant_q24 quant(.aclk,.aresetn,
      .s_acc(acc_fifo[rp][((index+q)*32)+:32]),.s_multiplier(multiplier),
      .s_valid(queued_groups!=0),.s_ready(q_ready[q]),.m_value(value),
      .m_valid(q_valid[q]),.m_ready(m_ready));
    assign quant_data[q*8+:8]=(relu_pipe[4]&&value[7])?8'd0:value;
  end
  // ReLU needs its own pipeline bit, aligned with the Q24 five-stage path.
  always_ff @(posedge aclk)begin
    if(!aresetn)begin rp<=0;wp<=0;queued_groups<=0;index<=0;token_valid<=0;relu_pipe<=0;end
    else begin
      case({push,pop})
        2'b10:queued_groups<=queued_groups+1;
        2'b01:queued_groups<=queued_groups-1;
        default:;
      endcase
      if(push)begin
        acc_fifo[wp]<=array_acc;meta_fifo[wp]<=array_meta;overflow_fifo[wp]<=array_overflow;
        wp<=wp==DEPTH-1?0:wp+1;
      end
      if(quant_accept)begin
        if(index==LANES-4)begin index<=0;rp<=rp==DEPTH-1?0:rp+1;end
        else index<=index+4;
      end
      if(quant_advance)begin
        token_valid<={token_valid[3:0],queued_groups!=0};
        relu_pipe<={relu_pipe[3:0],relu};
        if(queued_groups!=0)token_pipe[0]<=token;
        for(int j=1;j<5;j++)if(token_valid[j-1])token_pipe[j]<=token_pipe[j-1];
      end
    end
  end
  initial if(DEPTH<2)$fatal(1,"conv_compute FIFO depth must be >=2");
endmodule
