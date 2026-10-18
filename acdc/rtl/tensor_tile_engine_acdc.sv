// Non-convolution operators over the convolution engine's shared feature RAM.
// op=0: LUT map, 1: signed maxpool2 then LUT, 2: nearest upsample2 then LUT,
// 3: concat(skip, upsample2(deep)), with an independent LUT for each branch.
// Concat input: skip banks first, deep banks at ceil(split/4)*H*W per-bank.
module tensor_tile_engine_acdc (
  input logic aclk, aresetn, start,
  input logic [1:0] cfg_op,
  input logic [15:0] cfg_h, cfg_w,
  input logic [7:0] cfg_channels, cfg_split,
  output logic busy, done,
  output logic [2:0] error,
  input logic lut_we,
  input logic [8:0] lut_addr,
  input logic [7:0] lut_data,
  output logic input_re,
  output logic [14:0] input_addr,
  input logic [31:0] input_data,
  input logic input_valid,
  output logic output_we,
  output logic [2:0] output_bank,
  output logic [12:0] output_addr,
  output logic [7:0] output_data
);
  typedef enum logic [2:0] {IDLE, CFG_PLANE, CFG_SIZE, CFG_CHECK, STREAM, DRAIN} state_t;
  state_t state;
  (* ram_style="distributed" *) logic [7:0] lut [512];
  logic [1:0] op,sample;
  logic [8:0] ih,iw,oh,ow,x,y,source_width;
  logic [15:0] in_plane,out_plane,deep_base,source_plane;
  logic [15:0] source_channel_base,source_row,source_pixel,read_address;
  logic [15:0] output_group_base,output_pixel;
  logic [7:0] channels,split,channel,local_channel;
  logic branch;
  logic [31:0] cfg_plane,cfg_out_plane,cfg_input_size,cfg_output_size,cfg_deep_base,cfg_deep_size;
  logic setup_valid;
  logic [7:0] cfg_groups,cfg_split_groups,cfg_deep_groups,cfg_output_groups;
  logic req_valid,req_branch,req_last;
  logic [1:0] req_lane,req_sample;
  logic [2:0] req_bank,mapped_bank;
  logic [12:0] req_address,mapped_address;
  logic signed [7:0] selected_input,pool_max,pool_next;
  logic data_valid,data_branch,data_last;
  logic [1:0] data_lane,data_sample;
  logic [2:0] data_bank;
  logic [12:0] data_address;
  logic [31:0] data_q;
  logic [7:0] mapped;
  logic mapped_valid,mapped_last;
  logic lookup_valid,lookup_last;
  logic [8:0] lookup_index;
  logic [2:0] lookup_bank;
  logic [12:0] lookup_address;
  logic [15:0] next_channel_base,row_step,pixel_step;
  logic switch_branch,finish_pixel,last_pixel,last_output;

  assign busy=state!=IDLE;
  assign input_re=state==STREAM;
  assign input_addr=read_address[14:0];
  assign output_we=mapped_valid;
  assign output_bank=mapped_bank;
  assign output_addr=mapped_address;
  assign output_data=mapped;
  assign selected_input=$signed(data_q[data_lane*8+:8]);
  assign pool_next=data_sample==0 || selected_input>pool_max ? selected_input : pool_max;
  assign finish_pixel=op!=1 || sample==3;
  assign last_pixel=x==ow-1 && y==oh-1;
  assign last_output=last_pixel && channel==channels-1;
  assign switch_branch=op==3 && !branch && channel+8'd1==split;
  assign next_channel_base=switch_branch ? deep_base :
    local_channel[1:0]==3 ? source_channel_base+source_plane : source_channel_base;
  assign row_step=op==1 ? {6'd0,source_width,1'b0} :
    (op==2 || branch) ? (y[0] ? {7'd0,source_width} : 16'd0) : {7'd0,source_width};
  assign pixel_step=op==1 ? 16'd2 : (op==2 || branch) ? (x[0] ? 16'd1 : 16'd0) : 16'd1;

  always_ff @(posedge aclk) begin
    if (lut_we && !busy) lut[lut_addr]<=lut_data;
  end

  // One read request per cycle. Pool consumes four requests per output;
  // MAP/nearest/concat produce one output per cycle after the RAM/LUT pipeline.
  // Coordinates advance with additions only; shape multipliers run at setup.
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      state<=IDLE; done<=0; error<=0; req_valid<=0; lookup_valid<=0; mapped_valid<=0;
    end else begin
      done<=0;
      req_valid<=input_re;
      lookup_valid<=0;
      mapped_valid<=lookup_valid;
      data_valid<=0;
      // Separate BRAM selection/pool comparison from distributed-LUT lookup.
      // This adds latency, but still accepts one completed pixel each cycle.
      if (lookup_valid) begin
        mapped<=lut[lookup_index]; mapped_last<=lookup_last;
        mapped_bank<=lookup_bank; mapped_address<=lookup_address;
      end
      if (input_re) begin
        req_lane<=local_channel[1:0]; req_sample<=sample; req_branch<=branch;
        req_bank<=channel[2:0]; req_address<=output_group_base+output_pixel;
        req_last<=last_output && finish_pixel;
      end
      case(state)
        IDLE: if (start) begin
          error<=0;
          setup_valid<=cfg_channels>0 && cfg_channels<=192 && cfg_h>0 && cfg_w>0 &&
            cfg_h <= (cfg_op==2 ? 128 : 256) && cfg_w <= (cfg_op==2 ? 128 : 256) &&
            !((cfg_op==1 || cfg_op==3) && (cfg_h[0] || cfg_w[0])) &&
            !(cfg_op==3 && (cfg_split==0 || cfg_split>=cfg_channels));
          op<=cfg_op; ih<=cfg_h[8:0]; iw<=cfg_w[8:0];
          oh<=cfg_op==1 ? cfg_h>>1 : cfg_op==2 ? cfg_h<<1 : cfg_h;
          ow<=cfg_op==1 ? cfg_w>>1 : cfg_op==2 ? cfg_w<<1 : cfg_w;
          channels<=cfg_channels; split<=cfg_split;
          cfg_groups<=({1'b0,cfg_channels}+8'd3)>>2;
          cfg_split_groups<=({1'b0,cfg_split}+8'd3)>>2;
          cfg_deep_groups<=({1'b0,cfg_channels}-{1'b0,cfg_split}+8'd3)>>2;
          cfg_output_groups<=({1'b0,cfg_channels}+8'd7)>>3;
          state<=CFG_PLANE;
        end
        CFG_PLANE: begin
          cfg_plane<={23'd0,ih}*{23'd0,iw};
          cfg_out_plane<={23'd0,oh}*{23'd0,ow};
          state<=CFG_SIZE;
        end
        CFG_SIZE: begin
          cfg_input_size<=cfg_groups*cfg_plane;
          cfg_deep_base<=cfg_split_groups*cfg_plane;
          cfg_deep_size<=cfg_deep_groups*(cfg_plane>>2);
          cfg_output_size<=cfg_output_groups*cfg_out_plane;
          state<=CFG_CHECK;
        end
        CFG_CHECK: begin
          if (!setup_valid || (op==3 ? cfg_deep_base+cfg_deep_size>32768 : cfg_input_size>32768) || cfg_output_size>8192) begin
            done<=1; error<=1; state<=IDLE;
          end else begin
            in_plane<=cfg_plane[15:0]; out_plane<=cfg_out_plane[15:0]; deep_base<=cfg_deep_base[15:0];
            source_plane<=cfg_plane[15:0]; source_width<=iw;
            channel<=0; local_channel<=0; branch<=0; x<=0; y<=0; sample<=0;
            source_channel_base<=0; source_row<=0; source_pixel<=0; read_address<=0;
            output_group_base<=0; output_pixel<=0; state<=STREAM;
          end
        end
        STREAM: begin
          if (!finish_pixel) begin
            sample<=sample+1'b1;
            read_address<=sample==1 ? read_address+source_width-16'd1 : read_address+16'd1;
          end else begin
            sample<=0;
            if (last_output) state<=DRAIN;
            else if (last_pixel) begin
              x<=0; y<=0; channel<=channel+1'b1; output_pixel<=0;
              if (channel[2:0]==7) output_group_base<=output_group_base+out_plane;
              source_channel_base<=next_channel_base; source_row<=next_channel_base;
              source_pixel<=next_channel_base; read_address<=next_channel_base;
              local_channel<=switch_branch ? 8'd0 : local_channel+1'b1;
              if (switch_branch) begin branch<=1; source_width<=iw>>1; source_plane<=in_plane>>2; end
            end else begin
              output_pixel<=output_pixel+1'b1;
              if (x==ow-1) begin
                x<=0; y<=y+1'b1;
                source_row<=source_row+row_step;
                source_pixel<=source_row+row_step; read_address<=source_row+row_step;
              end else begin
                x<=x+1'b1;
                source_pixel<=source_pixel+pixel_step; read_address<=source_pixel+pixel_step;
              end
            end
          end
        end
        DRAIN: ;
        default: begin done<=1; error<=3; state<=IDLE; end
      endcase
      // The shared BRAM's output word is captured before the lane select,
      // pool compare and LUT index logic; tags ride the same stage so the
      // wide BRAM route never feeds arithmetic directly.
      if (req_valid && (state==STREAM || state==DRAIN) && !input_valid) begin
        done<=1; error<=3; state<=IDLE; lookup_valid<=0; mapped_valid<=0; data_valid<=0;
      end else begin
        if (req_valid && (state==STREAM || state==DRAIN)) begin
          data_valid<=1; data_q<=input_data;
          data_lane<=req_lane; data_sample<=req_sample; data_branch<=req_branch;
          data_last<=req_last; data_bank<=req_bank; data_address<=req_address;
        end
        if (data_valid && (state==STREAM || state==DRAIN)) begin
          pool_max<=pool_next;
          if (op!=1 || data_sample==3) begin
            lookup_index<={data_branch,(op==1 ? pool_next[7:0] : selected_input[7:0])};
            lookup_valid<=1; lookup_last<=data_last;
            lookup_bank<=data_bank; lookup_address<=data_address;
          end
        end
      end
      if (mapped_valid && mapped_last) begin done<=1; state<=IDLE; end
    end
  end
endmodule
