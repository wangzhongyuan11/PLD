// Halo-tile convolution. All memories are synchronous-read byte banks.
// Host/controller writes only while idle. Input already includes global halo;
// local convolution is stride 1, padding 0. Output is requantized, before ReLU.
module conv_tile_engine (
  input logic aclk, aresetn,
  input logic start,
  input logic [15:0] cfg_in_h, cfg_in_w,
  input logic [6:0] cfg_ci, cfg_co,
  input logic cfg_kernel3,
  input logic cfg_fused_concat,
  input logic [6:0] cfg_split,
  output logic busy, done,
  output logic [2:0] error,
  input logic input_we,
  input logic [1:0] input_bank,
  input logic [14:0] input_addr,
  input logic [7:0] input_data,
  input logic weight_we,
  input logic [4:0] weight_bank,
  input logic [10:0] weight_addr,
  input logic [7:0] weight_data,
  input logic param_we, param_multiplier,
  input logic [5:0] param_channel,
  input logic [31:0] param_data,
  input logic output_re,
  input logic [2:0] output_bank,
  input logic [12:0] output_addr,
  output logic [7:0] output_data,
  output logic output_valid,
  // Shared feature RAM access for non-convolution PL operators. Controller
  // must arbitrate loaders/readback against this port and the convolution.
  input logic aux_input_re,
  input logic [14:0] aux_input_addr,
  output logic [31:0] aux_input_data,
  output logic aux_input_valid,
  input logic aux_output_we,
  input logic [2:0] aux_output_bank,
  input logic [12:0] aux_output_addr,
  input logic [7:0] aux_output_data
);
  typedef enum logic [3:0] {IDLE, CFG_PLANE, CFG_SIZE, CFG_CHECK, ISSUE, WAIT_MAC, QUANT, DRAIN, NEXT_PIXEL} state_t;
  state_t state;
  logic [15:0] iw, oh, ow, x, y;
  logic [15:0] in_plane, out_plane;
  logic [6:0] ci, co, out_channel;
  logic [4:0] ci_group, ci_groups;
  logic [1:0] kx, ky, km1;
  logic [14:0] pixel_base, channel_offset, row_offset;
  logic [14:0] read_input_addr;
  logic [10:0] read_weight_addr, weight_base, weights_per_group;
  logic [12:0] out_pixel, out_group_base, write_output_addr;
  logic [3:0] lanes, quant_sent, quant_received;
  logic read_valid, read_first, read_last;
  logic [3:0] read_lane_mask;
  logic [31:0] input_words, mac_input;
  logic [255:0] weight_words, mac_weights, mac_bias, mac_acc, held_acc;
  logic [31:0] mac_input_q;
  logic [255:0] mac_weights_q,mac_bias_q;
  logic mac_valid_q,mac_first_q,mac_last_q;
  logic [31:0] bias [64];
  logic [31:0] multiplier [64];
  logic mac_ready, mac_valid, mac_overflow;
  logic quant_ready, quant_valid;
  logic signed [7:0] quant_value;
  logic [31:0] quant_multiplier;
  logic [31:0] cfg_input_plane, cfg_output_plane, cfg_input_occupancy, cfg_output_occupancy;
  logic [15:0] setup_h;
  logic setup_valid;
  logic [4:0] cfg_co_groups;
  // Fused decoder-input addressing (mode 6). Deep channels are stored as a
  // compact half-resolution patch; the engine folds nearest-2 upsample into
  // the read address so skip/deep branches share the normal MAC path.
  logic fused;
  logic [6:0] skip_groups, deep_groups;
  logic [15:0] deep_plane, deep_width, deep_base, deep_channel_base, deep_row0, deep_row;
  logic [7:0] deep_col;
  logic deep_group;
  logic [15:0] deep_read_addr;
  logic shared_input_re, shared_output_we, conv_output_we;
  logic [14:0] shared_input_addr;
  logic [2:0] shared_output_bank;
  logic [12:0] shared_output_addr;
  logic [7:0] shared_output_data;

  assign busy = state != IDLE;
  assign read_input_addr = pixel_base + channel_offset + row_offset + kx;
  assign deep_read_addr = deep_channel_base + deep_row + {8'd0,deep_col};
  assign write_output_addr = out_group_base + out_pixel;
  // Arbitrate BEFORE the RAM statement: multiple syntactic read/write
  // accesses infer extra memory ports and can silently fall back to LUT RAM.
  assign shared_input_re = state==ISSUE || (aux_input_re && !busy);
  assign shared_input_addr = state==ISSUE ?
    (fused && deep_group ? deep_read_addr[14:0] : read_input_addr) : aux_input_addr;
  assign conv_output_we = quant_valid && (state==QUANT || state==DRAIN);
  assign shared_output_we = conv_output_we || (aux_output_we && !busy);
  assign shared_output_bank = conv_output_we ? quant_received[2:0] : aux_output_bank;
  assign shared_output_addr = conv_output_we ? write_output_addr : aux_output_addr;
  assign shared_output_data = conv_output_we ? quant_value : aux_output_data;

  for (genvar b=0; b<4; b++) begin : input_banks
    (* ram_style="block" *) logic [7:0] mem [32768];
    always_ff @(posedge aclk) begin
      if (input_we && !busy && input_bank==b) mem[input_addr] <= input_data;
      if (shared_input_re) input_words[b*8+:8] <= mem[shared_input_addr];
    end
  end
  for (genvar b=0; b<32; b++) begin : weight_banks
    (* ram_style="block" *) logic [7:0] mem [2048];
    always_ff @(posedge aclk) begin
      if (weight_we && !busy && weight_bank==b) mem[weight_addr] <= weight_data;
      if (state==ISSUE) weight_words[b*8+:8] <= mem[read_weight_addr];
    end
  end
  logic [7:0] output_words [8];
  logic [2:0] output_bank_q;
  for (genvar b=0; b<8; b++) begin : output_banks
    (* ram_style="block" *) logic [7:0] mem [8192];
    always_ff @(posedge aclk) begin
      if (shared_output_we && shared_output_bank==b) mem[shared_output_addr] <= shared_output_data;
      if (output_re && !busy) output_words[b] <= mem[output_addr];
    end
  end
  assign output_data = output_words[output_bank_q];
  assign aux_input_data = input_words;
  always_ff @(posedge aclk) begin
    if (param_we && !busy) begin
      if (param_multiplier) multiplier[param_channel] <= param_data;
      else bias[param_channel] <= param_data;
    end
    if (output_re && !busy) output_bank_q <= output_bank;
  end
  always_ff @(posedge aclk) begin
    if (!aresetn) begin output_valid <= 0; aux_input_valid <= 0; end
    else begin
      output_valid <= output_re && !busy;
      aux_input_valid <= aux_input_re && !busy;
    end
  end

  always_comb begin
    mac_input = 0;
    mac_weights = 0;
    mac_bias = 0;
    for (int i=0; i<4; i++)
      if (read_lane_mask[i]) mac_input[i*8+:8] = input_words[i*8+:8];
    for (int o=0; o<8; o++) begin
      if (o < lanes) begin
        mac_bias[o*32+:32] = bias[out_channel+o];
        for (int i=0; i<4; i++)
          if (read_lane_mask[i]) mac_weights[o*32+i*8+:8] = weight_words[o*32+i*8+:8];
      end
    end
    quant_multiplier = 0;
    if (quant_sent < lanes) quant_multiplier = multiplier[out_channel+quant_sent];
  end
  // Break the BRAM/lane-mask -> DSP multiplier path. The issue loop never
  // starts a second dot product until the first MAC/requant result is drained.
  always_ff @(posedge aclk) begin
    if (!aresetn) mac_valid_q<=0;
    else begin
      mac_valid_q<=read_valid;
      if (read_valid) begin
        mac_input_q<=mac_input; mac_weights_q<=mac_weights; mac_bias_q<=mac_bias;
        mac_first_q<=read_first; mac_last_q<=read_last;
      end
    end
  end
  conv_mac_array mac (
    .aclk, .aresetn, .s_input(mac_input_q), .s_weights(mac_weights_q), .s_bias(mac_bias_q),
    .s_first(mac_first_q), .s_last(mac_last_q), .s_valid(mac_valid_q), .s_ready(mac_ready),
    .m_acc(mac_acc), .m_overflow(mac_overflow), .m_valid(mac_valid), .m_ready(state==WAIT_MAC)
  );
  requant_q24 requant (
    .aclk, .aresetn, .s_acc(held_acc[quant_sent[2:0]*32+:32]),
    .s_multiplier(quant_multiplier), .s_valid(state==QUANT && quant_sent<lanes),
    .s_ready(quant_ready), .m_value(quant_value), .m_valid(quant_valid), .m_ready(1'b1)
  );

  // Only control/valid state needs reset; use a synchronous process so
  // unreset address/data registers infer clock enables, not async set/reset.
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      state <= IDLE;
      done <= 0;
      error <= 0;
      read_valid <= 0;
    end else begin
      done <= 0;
      read_valid <= state==ISSUE;
      if (state==ISSUE) begin
        read_first <= ci_group==0 && ky==0 && kx==0;
        read_last <= ci_group==ci_groups-1 && ky==km1 && kx==km1;
        for (int i=0; i<4; i++) read_lane_mask[i] <= ({ci_group,2'b00}+i) < ci;
      end
      case (state)
        IDLE: if (start) begin
          error <= 0;
          setup_valid<=cfg_ci>0 && cfg_ci<=96 && cfg_co>0 && cfg_co<=64 &&
            cfg_in_h >= (cfg_kernel3 ? 3 : 1) && cfg_in_w >= (cfg_kernel3 ? 3 : 1) &&
            cfg_in_h <= (cfg_kernel3 ? 258 : 256) && cfg_in_w <= (cfg_kernel3 ? 258 : 256) &&
            (!cfg_fused_concat || (cfg_kernel3 && !cfg_in_h[0] && !cfg_in_w[0] &&
              cfg_split>0 && cfg_split<cfg_ci && cfg_split[1:0]==0 && cfg_ci[1:0]==cfg_split[1:0]));
          setup_h<=cfg_in_h; iw<=cfg_in_w;
          oh<=cfg_in_h-(cfg_kernel3 ? 16'd2 : 16'd0);
          ow<=cfg_in_w-(cfg_kernel3 ? 16'd2 : 16'd0);
          ci<=cfg_ci; co<=cfg_co;
          ci_groups<=({1'b0,cfg_ci}+8'd3)>>2;
          cfg_co_groups<=({1'b0,cfg_co}+8'd7)>>3;
          fused<=cfg_fused_concat;
          skip_groups<=({1'b0,cfg_split}+8'd3)>>2;
          deep_groups<=({1'b0,cfg_ci}-{1'b0,cfg_split}+8'd3)>>2;
          km1<=cfg_kernel3 ? 2 : 0;
          state<=CFG_PLANE;
        end
        // Register between dimensional products. These paths start at the
        // top-level AXI configuration registers in the complete BD.
        CFG_PLANE: begin
          cfg_input_plane<={23'd0,setup_h[8:0]}*{23'd0,iw[8:0]};
          cfg_output_plane<={23'd0,oh[8:0]}*{23'd0,ow[8:0]};
          // Compact deep patch: deep_width=(iw/2+1) columns; one deep row
          // advances by deep_width, a deep channel group by deep_plane.
          deep_plane<=(({7'd0,setup_h[8:0]}>>1)+16'd1)*(({7'd0,iw[8:0]}>>1)+16'd1);
          deep_width<=({7'd0,iw[8:0]}>>1)+16'd1;
          state<=CFG_SIZE;
        end
        CFG_SIZE: begin
          cfg_input_occupancy<=fused ?
            skip_groups*cfg_input_plane+deep_groups*{16'd0,deep_plane} :
            cfg_input_plane*ci_groups;
          cfg_output_occupancy<=cfg_output_plane*cfg_co_groups;
          in_plane<=cfg_input_plane[15:0]; out_plane<=cfg_output_plane[15:0];
          deep_base<=skip_groups*cfg_input_plane;
          state<=CFG_CHECK;
        end
        CFG_CHECK: begin
          if (!setup_valid || cfg_input_occupancy>32768 || cfg_output_occupancy>8192) begin
            error <= 1;
            done <= 1;
            state <= IDLE;
          end else begin
            weights_per_group <= km1==2 ? ci_groups*11'd9 : {6'd0,ci_groups};
            lanes <= co>=8 ? 8 : {1'b0,co[2:0]};
            x<=0; y<=0; out_channel<=0; out_pixel<=0; out_group_base<=0;
            pixel_base<=0; channel_offset<=0; row_offset<=0;
            ci_group<=0; kx<=0; ky<=0; weight_base<=0; read_weight_addr<=0;
            deep_group<=0; deep_row0<=0; deep_row<=0; deep_col<=0;
            deep_channel_base<=deep_base;
            state<=ISSUE;
          end
        end
        ISSUE: begin
          read_weight_addr <= read_weight_addr+1'b1;
          if (kx!=km1) begin
            kx<=kx+1'b1;
            if (fused) deep_col<=deep_col+(x[0]==kx[0]);
          end
          else begin
            kx<=0;
            if (ky!=km1) begin
              ky<=ky+1'b1; row_offset<=row_offset+iw;
              if (fused) begin
                deep_row<=deep_row+((y[0]==ky[0])?deep_width:16'd0);
                deep_col<=(x>>1)+{8'd0,x[0]};
              end
            end
            else begin
              ky<=0; row_offset<=0;
              if (ci_group!=ci_groups-1) begin
                ci_group<=ci_group+1'b1; channel_offset<=channel_offset+in_plane;
                // Classify the NEXT group from the group counter itself; a
                // stale deep_group from the previous pixel must never steer
                // skip-group reads or advance the deep channel base.
                if (fused) begin
                  if (ci_group+7'd1==skip_groups) begin
                    deep_group<=1; deep_channel_base<=deep_base;
                  end else if (ci_group+7'd1>skip_groups) begin
                    deep_group<=1; deep_channel_base<=deep_channel_base+deep_plane;
                  end else deep_group<=0;
                  deep_row<=deep_row0; deep_col<=(x>>1)+{8'd0,x[0]};
                end
              end else state<=WAIT_MAC;
            end
          end
        end
        WAIT_MAC: if (mac_valid) begin
          if (mac_overflow) begin error<=2; done<=1; state<=IDLE; end
          else begin
            held_acc<=mac_acc; quant_sent<=0; quant_received<=0; state<=QUANT;
          end
        end
        QUANT, DRAIN: begin
          if (state==QUANT && quant_ready) begin
            quant_sent<=quant_sent+1'b1;
            if (quant_sent==lanes-1) state<=DRAIN;
          end
          if (quant_valid) begin
            quant_received<=quant_received+1'b1;
            if (quant_received==lanes-1) state<=NEXT_PIXEL;
          end
        end
        NEXT_PIXEL: begin
          ci_group<=0; channel_offset<=0; row_offset<=0; kx<=0; ky<=0;
          // Every new pixel restarts in skip group 0; deep_row/deep_col and
          // deep_channel_base are re-established at the first deep switch.
          deep_group<=0;
          read_weight_addr<=weight_base;
          if (x==ow-1 && y==oh-1) begin
            if (out_channel+8>=co) begin done<=1; state<=IDLE; end
            else begin
              out_channel<=out_channel+8; out_group_base<=out_group_base+out_plane;
              lanes <= co-out_channel-8>=8 ? 8 : co-out_channel-8;
              weight_base<=weight_base+weights_per_group;
              read_weight_addr<=weight_base+weights_per_group;
              x<=0; y<=0; pixel_base<=0; out_pixel<=0; state<=ISSUE;
              if (fused) deep_row0<=0;
            end
          end else begin
            out_pixel<=out_pixel+1'b1;
            if (x==ow-1) begin
              x<=0; y<=y+1'b1; pixel_base<=pixel_base+{14'd0,km1}+1'b1;
              if (fused) deep_row0<=deep_row0+((y[0]==0)?deep_width:16'd0);
            end
            else begin x<=x+1'b1; pixel_base<=pixel_base+1'b1; end
            state<=ISSUE;
          end
        end
        default: begin state<=IDLE; error<=3; done<=1; end
      endcase
    end
  end
endmodule
