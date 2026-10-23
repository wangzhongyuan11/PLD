// Drop-in CAM2-shape tile compute: true dual-port reads combine two groups of
// four Ci and two groups of eight Co. Weight RAM is replicated (same writes)
// for four read addresses; external NCHW/OIHW byte and bank layouts stay exact.
module conv_tile_engine_acdc_128 (
  input logic aclk,aresetn,start,
  input logic [15:0] cfg_in_h,cfg_in_w,
  input logic [7:0] cfg_ci,
  input logic [6:0] cfg_co,
  input logic cfg_relu,cfg_kernel3,cfg_fused_concat,
  input logic [7:0] cfg_split,
  output logic busy,done,
  output logic [2:0] error,
  input logic input_we,
  input logic [1:0] input_bank,
  input logic [14:0] input_addr,
  input logic [7:0] input_data,
  input logic weight_we,
  input logic [4:0] weight_bank,
  input logic [10:0] weight_addr,
  input logic [7:0] weight_data,
  input logic param_we,param_multiplier,
  input logic [5:0] param_channel,
  input logic [31:0] param_data,
  input logic output_re,
  input logic [2:0] output_bank,
  input logic [12:0] output_addr,
  output logic [7:0] output_data,
  output logic output_valid,
  input logic aux_input_re,
  input logic [14:0] aux_input_addr,
  output logic [31:0] aux_input_data,
  output logic aux_input_valid,
  input logic aux_output_we,
  input logic [2:0] aux_output_bank,
  input logic [12:0] aux_output_addr,
  input logic [7:0] aux_output_data
);
  typedef enum logic [3:0] {IDLE,SETUP,SIZE,CHECK,ISSUE,WAIT_MAC,QUANT,DRAIN,NEXT} st_t;
  st_t state;
  logic [15:0] iw,oh,ow,in_plane,out_plane,x,y,pixel;
  logic [7:0] ci;
  logic [6:0] co,out_ch;
  logic [5:0] ci_groups,ci_group;
  logic [1:0] kx,ky,km1;
  logic relu;
  logic [3:0] ks;
  logic [15:0] weight_stride;
  logic [14:0] pixel_base,channel_offset,row_offset;
  logic [10:0] weight_current,weight_group_base;
  logic [12:0] output_group_base;
  logic [4:0] lanes,quant_sent,quant_received;
  logic [31:0] bias [64],multiplier [64];
  logic [14:0] ia,ib;
  logic [14:0] input_porta_addr;
  logic [10:0] wa,wb,wc,wd;
  logic [10:0] weight_porta0_addr,weight_porta1_addr;
  logic [31:0] input_a,input_b;
  logic [255:0] weight_a,weight_b,weight_c,weight_d;
  logic [63:0] mac_input;
  logic [1023:0] mac_weights;
  logic [511:0] mac_bias,mac_acc,held_acc;
  logic [63:0] issue_input;
  logic [1023:0] issue_weight;
  logic [511:0] issue_bias;
  logic read_valid,read_first,read_last,issue_valid,issue_first,issue_last;
  logic [7:0] lane_mask;
  logic mac_ready,mac_valid,mac_overflow;
  logic q_ready,q_valid;
  logic signed [7:0] q_value;
  logic [31:0] q_mult;
  logic [2:0] obank,read_obank;
  logic [12:0] oaddr;
  logic [7:0] odata,out_words [8];
  logic owe,conv_write;
  logic [31:0] setup_input_occupancy,setup_output_occupancy,setup_weight_occupancy;
  assign busy=state!=IDLE;
  assign ia=pixel_base+channel_offset+row_offset+kx;
  assign ib=ia+in_plane;
  assign wa=weight_current;
  assign wb=wa+ks;
  assign wc=wa+weight_stride;
  assign wd=wc+ks;
  // Write/read selection happens BEFORE the memory statement. Three syntactic
  // addresses otherwise infer duplicated BRAM for a nominal two-port array.
  assign input_porta_addr=(input_we&&!busy)?input_addr:state==ISSUE?ia:aux_input_addr;
  assign weight_porta0_addr=(weight_we&&!busy)?weight_addr:wa;
  assign weight_porta1_addr=(weight_we&&!busy)?weight_addr:wc;
  assign conv_write=q_valid&&(state==QUANT||state==DRAIN);
  assign owe=conv_write||(aux_output_we&&!busy);
  assign obank=conv_write ? quant_received[2:0] : aux_output_bank;
  assign oaddr=conv_write ? (output_group_base+(quant_received[3]?out_plane:0)+pixel) : aux_output_addr;
  assign odata=conv_write ? ((relu&&q_value[7])?8'd0:q_value) : aux_output_data;
  assign output_data=out_words[read_obank];
  assign aux_input_data=input_a;
  for(genvar b=0;b<4;b++)begin:inputs
    (* ram_style="block" *) logic [7:0] mem [32768];
    always_ff @(posedge aclk)begin
      // Port A arbitrates load / conv-first-group / tensor reads.
      if((input_we&&!busy&&input_bank==b)||state==ISSUE||(aux_input_re&&!busy))begin
        if(input_we&&!busy&&input_bank==b)mem[input_porta_addr]<=input_data;
        input_a[b*8+:8]<=mem[input_porta_addr];
      end
    end
    // Explicit independent port processes match the supported TDP template.
    always_ff @(posedge aclk)if(state==ISSUE)input_b[b*8+:8]<=mem[ib];
  end
  for(genvar b=0;b<32;b++)begin:weights
    (* ram_style="block" *) logic [7:0] mem0 [2048];
    (* ram_style="block" *) logic [7:0] mem1 [2048];
    always_ff @(posedge aclk)begin
      if((weight_we&&!busy&&weight_bank==b)||state==ISSUE)begin
        if(weight_we&&!busy&&weight_bank==b)mem0[weight_porta0_addr]<=weight_data;
        weight_a[b*8+:8]<=mem0[weight_porta0_addr];
      end
    end
    always_ff @(posedge aclk)if(state==ISSUE)weight_b[b*8+:8]<=mem0[wb];
    always_ff @(posedge aclk)begin
      if((weight_we&&!busy&&weight_bank==b)||state==ISSUE)begin
        if(weight_we&&!busy&&weight_bank==b)mem1[weight_porta1_addr]<=weight_data;
        weight_c[b*8+:8]<=mem1[weight_porta1_addr];
      end
    end
    always_ff @(posedge aclk)if(state==ISSUE)weight_d[b*8+:8]<=mem1[wd];
  end
  for(genvar b=0;b<8;b++)begin:outputs
    (* ram_style="block" *) logic [7:0] mem [8192];
    always_ff @(posedge aclk)begin
      if(owe&&obank==b)mem[oaddr]<=odata;
      if(output_re&&!busy)out_words[b]<=mem[output_addr];
    end
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)begin output_valid<=0;aux_input_valid<=0;end
    else begin output_valid<=output_re&&!busy;aux_input_valid<=aux_input_re&&!busy;end
    if(output_re&&!busy)read_obank<=output_bank;
    if(param_we&&!busy)begin
      if(param_multiplier)multiplier[param_channel]<=param_data;
      else bias[param_channel]<=param_data;
    end
  end
  always_comb begin
    mac_input=0;mac_weights=0;mac_bias=0;
    for(int i=0;i<8;i++)if(lane_mask[i])
      mac_input[i*8+:8]=(i<4)?input_a[i*8+:8]:input_b[(i-4)*8+:8];
    for(int o=0;o<16;o++)if(o<lanes)begin
      mac_bias[o*32+:32]=bias[out_ch+o];
      for(int i=0;i<8;i++)if(lane_mask[i])begin
        if(o<8&&i<4)mac_weights[(o*8+i)*8+:8]=weight_a[(o*4+i)*8+:8];
        else if(o<8)mac_weights[(o*8+i)*8+:8]=weight_b[(o*4+i-4)*8+:8];
        else if(i<4)mac_weights[(o*8+i)*8+:8]=weight_c[((o-8)*4+i)*8+:8];
        else mac_weights[(o*8+i)*8+:8]=weight_d[((o-8)*4+i-4)*8+:8];
      end
    end
    q_mult=0;
    if(quant_sent<lanes)q_mult=multiplier[out_ch+quant_sent];
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)issue_valid<=0;
    else begin
      issue_valid<=read_valid;
      if(read_valid)begin
        issue_input<=mac_input;issue_weight<=mac_weights;issue_bias<=mac_bias;
        issue_first<=read_first;issue_last<=read_last;
      end
    end
  end
  conv_mac_128 mac(.aclk,.aresetn,.s_input(issue_input),.s_weights(issue_weight),.s_bias(issue_bias),
    .s_first(issue_first),.s_last(issue_last),.s_valid(issue_valid),.s_ready(mac_ready),
    .m_acc(mac_acc),.m_overflow(mac_overflow),.m_valid(mac_valid),.m_ready(state==WAIT_MAC));
  requant_q24 quant(.aclk,.aresetn,.s_acc(held_acc[quant_sent[3:0]*32+:32]),.s_multiplier(q_mult),
    .s_valid(state==QUANT&&quant_sent<lanes),.s_ready(q_ready),.m_value(q_value),.m_valid(q_valid),.m_ready(1'b1));
  always_ff @(posedge aclk)begin
    if(!aresetn)begin state<=IDLE;done<=0;error<=0;read_valid<=0;end
    else begin
      done<=0;read_valid<=state==ISSUE;
      if(state==ISSUE)begin
        for(int i=0;i<8;i++)lane_mask[i]<=({ci_group,2'b00}+i)<ci;
        read_first<=ci_group==0&&ky==0&&kx==0;
        read_last<=ci_group+2>=ci_groups&&ky==km1&&kx==km1;
      end
      case(state)
        IDLE:if(start)begin
          error<=0;iw<=cfg_in_w;oh<=cfg_in_h-(cfg_kernel3?2:0);ow<=cfg_in_w-(cfg_kernel3?2:0);
          ci<=cfg_ci;co<=cfg_co;km1<=cfg_kernel3?2:0;ks<=cfg_kernel3?9:1;relu<=cfg_relu;
          ci_groups<=({1'b0,cfg_ci}+3)>>2;
          if(cfg_in_h<(cfg_kernel3?3:1)||cfg_in_w<(cfg_kernel3?3:1)||cfg_ci==0||cfg_ci>192||cfg_co==0||cfg_co>64||cfg_fused_concat)begin
            error<=1;done<=1;
          end else state<=SETUP;
        end
        SETUP:begin
          in_plane<=cfg_in_h*cfg_in_w;out_plane<=oh*ow;
          weight_stride<=ci_groups*ks;
          state<=SIZE;
        end
        SIZE:begin
          setup_input_occupancy<=ci_groups*in_plane;
          setup_output_occupancy<=((co+7)>>3)*out_plane;
          setup_weight_occupancy<=((co+7)>>3)*weight_stride;
          state<=CHECK;
        end
        CHECK:begin
          if(setup_input_occupancy>32768||setup_output_occupancy>8192||setup_weight_occupancy>2048||oh>256||ow>256)begin
            error<=1;done<=1;state<=IDLE;
          end else begin
            x<=0;y<=0;pixel<=0;out_ch<=0;ci_group<=0;ky<=0;kx<=0;lanes<=co<16?co:16;
            pixel_base<=0;channel_offset<=0;row_offset<=0;
            weight_current<=0;weight_group_base<=0;output_group_base<=0;state<=ISSUE;
          end
        end
        ISSUE:begin
          weight_current<=weight_current+1;
          if(kx<km1)kx<=kx+1;
          else begin
            kx<=0;
            if(ky<km1)begin ky<=ky+1;row_offset<=row_offset+iw;end
            else begin
              ky<=0;row_offset<=0;
              if(ci_group+2<ci_groups)begin
                ci_group<=ci_group+2;channel_offset<=channel_offset+(in_plane<<1);
                weight_current<=weight_current+ks+1;
              end else state<=WAIT_MAC;
            end
          end
        end
        WAIT_MAC:if(mac_valid)begin
          if(mac_overflow)begin error<=2;done<=1;state<=IDLE;end
          else begin held_acc<=mac_acc;quant_sent<=0;quant_received<=0;state<=QUANT;end
        end
        QUANT,DRAIN:begin
          if(state==QUANT&&q_ready)begin quant_sent<=quant_sent+1;if(quant_sent==lanes-1)state<=DRAIN;end
          if(q_valid)begin quant_received<=quant_received+1;if(quant_received==lanes-1)state<=NEXT;end
        end
        NEXT:begin
          ci_group<=0;ky<=0;kx<=0;channel_offset<=0;row_offset<=0;weight_current<=weight_group_base;
          if(x+1<ow)begin x<=x+1;pixel<=pixel+1;pixel_base<=pixel_base+1;state<=ISSUE;end
          else if(y+1<oh)begin x<=0;y<=y+1;pixel<=pixel+1;pixel_base<=pixel_base+iw-ow+1;state<=ISSUE;end
          else if(out_ch+16<co)begin
            out_ch<=out_ch+16;lanes<=co-out_ch-16<16?co-out_ch-16:16;
            weight_group_base<=weight_group_base+(weight_stride<<1);
            weight_current<=weight_group_base+(weight_stride<<1);
            output_group_base<=output_group_base+(out_plane<<1);
            x<=0;y<=0;pixel<=0;pixel_base<=0;state<=ISSUE;
          end else begin done<=1;state<=IDLE;end
        end
        default:state<=IDLE;
      endcase
    end
  end
endmodule
