// ABI 1.2 command processor around the banked tile engines.
// Commands: 1=params, 2=input, 3=LUT, 4=run. Modes: 0=conv, 1=map,
// 2=pool, 3=upsample, 4=concat(skip, upsample(deep)). All packets are NCHW/
// OIHW externally; mode 5 maps input bytes through LUT0 while loading a
// convolution tile. Mode 6 additionally accepts a compact half-resolution
// deep patch after the skip patch and maps it through LUT1; the conv engine
// folds nearest-2 upsampling into its deep read addresses. Outputs stay raw.
module axis_tensor_accel_acdc (
  input logic aclk, aresetn,
  input logic [63:0] s_axis_tdata,
  input logic [7:0] s_axis_tkeep,
  input logic s_axis_tvalid, s_axis_tlast,
  output logic s_axis_tready,
  output logic [63:0] m_axis_tdata,
  output logic [7:0] m_axis_tkeep,
  output logic m_axis_tvalid, m_axis_tlast,
  input logic m_axis_tready,
  input logic [11:0] s_axi_awaddr,
  input logic [2:0] s_axi_awprot,
  input logic s_axi_awvalid,
  output logic s_axi_awready,
  input logic [31:0] s_axi_wdata,
  input logic [3:0] s_axi_wstrb,
  input logic s_axi_wvalid,
  output logic s_axi_wready,
  output logic [1:0] s_axi_bresp,
  output logic s_axi_bvalid,
  input logic s_axi_bready,
  input logic [11:0] s_axi_araddr,
  input logic [2:0] s_axi_arprot,
  input logic s_axi_arvalid,
  output logic s_axi_arready,
  output logic [31:0] s_axi_rdata,
  output logic [1:0] s_axi_rresp,
  output logic s_axi_rvalid,
  input logic s_axi_rready
);
  typedef enum logic [3:0] {IDLE,SHAPE1,SHAPE2,SHAPE3,CHECK,LOAD,
    CORE_START,CORE_WAIT,OUT_STREAM,OUT_DRAIN,FAULT,LOAD_DRAIN} state_t;
  state_t state;
  logic busy,done,soft_reset_q,core_resetn,start_command;
  logic [31:0] error_code;
  logic wr_en,rd_en;
  logic [11:0] wr_addr,rd_addr;
  logic [31:0] wr_data,rd_data;
  logic [3:0] wr_strb;
  logic [1:0] wr_resp,rd_resp;
  logic [31:0] relu_flags, weight_occupancy;
  logic [31:0] command,mode,height,width,ci,co,kernel,split,frame_id,job_id;
  logic [31:0] completed_frame,completed_job,cycles,compute_cycles;
  logic [31:0] expected_in,expected_out,received_bytes,sent_bytes,output_count;
  logic [31:0] input_remaining;
  logic expected_last_input;
  logic params_ready,input_ready,lut_ready;
  logic params_match,input_match,lut_match;
  logic [31:0] tag_ci,tag_co,tag_kernel;
  logic [31:0] tag_mode,tag_h,tag_w,tag_input_ci,tag_split;
  logic lut_dual;
  logic convolution_mode;
  (* ram_style="distributed" *) logic [7:0] input_lut [512];
  logic [7:0] convolution_input_byte;

  logic [31:0] plane,out_h,out_w,out_plane,input_occupancy,output_occupancy;
  logic [31:0] channel_product,weight_bytes,deep_base,deep_plane6,deep_group_bytes;
  logic [31:0] normal_input_bytes,skip_input_bytes,deep_input_bytes;
  logic [7:0] deep_channels;
  logic [6:0] ci_groups,co_groups,skip_groups,deep_groups;
  logic [7:0] result_channels;
  logic [3:0] kernel_square;
  logic [10:0] weights_per_group;
  logic check_shape,check_dims;
  logic [7:0] load_byte,result_byte;
  logic load_valid,load_last,load_ready,unpack_ready,unpack_error,write_byte;
  logic pack_ready;
  logic input_we,weight_we,param_we,lut_we;
  logic input_we_q;
  logic [1:0] input_bank_q;
  logic [14:0] input_address_q;
  logic [7:0] input_byte_q;
  logic [1:0] param_phase,param_byte;
  logic [5:0] param_channel;
  logic [31:0] param_word,param_next;
  logic [7:0] weight_o,weight_i;
  logic [3:0] weight_k;
  logic [10:0] weight_group_base,weight_input_base,weight_address;
  logic [7:0] input_channel,input_channel_limit;
  logic input_branch;
  logic [15:0] input_pixel,input_plane;
  logic [14:0] input_group_base,input_address;
  logic [7:0] output_channel;
  logic [15:0] output_pixel;
  logic [12:0] output_group_base,output_address;
  logic output_valid;
  logic [7:0] output_data;
  logic output_read_enable,output_read_pending,output_push,output_pop;
  logic [31:0] output_issued;
  logic [7:0] output_queue [4];
  logic [1:0] output_rptr,output_wptr;
  logic [2:0] output_queued;
  logic conv_busy,conv_done,tensor_busy,tensor_done;
  logic [2:0] conv_error,tensor_error;
  logic aux_input_re,aux_input_valid,aux_output_we;
  logic [14:0] aux_input_addr;
  logic [31:0] aux_input_data;
  logic [2:0] aux_output_bank;
  logic [12:0] aux_output_addr;
  logic [7:0] aux_output_data;

  function automatic logic [31:0] merged(input logic [31:0] old_value,new_value,input logic [3:0] strobe);
    for (int b=0;b<4;b++) merged[b*8+:8]=strobe[b] ? new_value[b*8+:8] : old_value[b*8+:8];
  endfunction
  function automatic logic [3:0] keep_count(input logic [7:0] keep);
    keep_count=0;
    for (int b=0;b<8;b++) keep_count=keep_count+keep[b];
  endfunction
  assign busy=state!=IDLE && state!=FAULT;
  always_ff @(posedge aclk) begin
    if (!aresetn) convolution_mode<=0;
    else if (state==IDLE && start_command) convolution_mode<=mode==0 || mode==5 || mode==6;
  end
  assign convolution_input_byte=mode==5 ? input_lut[load_byte] :
    mode==6 ? input_lut[{input_branch,load_byte}] : load_byte;
  always_ff @(posedge aclk) begin
    if (lut_we) input_lut[received_bytes[8:0]]<=load_byte;
  end
  assign start_command=wr_en && wr_resp==0 && wr_addr==12'h008 && wr_strb[0] && wr_data[0] && !wr_data[1];
  // Apply the accepted reset one cycle after the write commit. The write
  // itself still commits combinationally; the register keeps the wide
  // wr_addr/wr_data decode cone out of every engine's reset network.
  always_ff @(posedge aclk) begin
    if (!aresetn) soft_reset_q<=0;
    else soft_reset_q<=wr_en && wr_addr==12'h008 && wr_strb[0] && wr_data[1];
  end
  assign core_resetn=aresetn && !soft_reset_q && state!=FAULT;
  assign params_match=params_ready && ci==tag_ci && co==tag_co && kernel==tag_kernel;
  assign input_match=input_ready && mode==tag_mode && height==tag_h && width==tag_w && ci==tag_input_ci && split==tag_split;
  assign lut_match=lut_ready && ((mode!=4 && mode!=6) || lut_dual);
  axi_lite_endpoint control(.*);

  always_comb begin
    wr_resp=2'b10;
    if (wr_addr==12'h008) begin
      if (wr_strb[0] && wr_data[1]) wr_resp=0; // reset is accepted even while busy/faulted
      else if (!busy && state!=FAULT) wr_resp=0;
    end else if (!busy && state!=FAULT) begin
      case(wr_addr)
        12'h014,12'h018,12'h01c,12'h020,12'h024,12'h028,12'h02c,12'h030,12'h034,12'h038,12'h05c: wr_resp=0;
        default: wr_resp=2'b10;
      endcase
    end
    rd_resp=0; rd_data=0;
    case(rd_addr)
      12'h000: rd_data=32'h41434433;
      12'h004: rd_data=32'h00030000;
      12'h008: rd_data=0;
      12'h00c: rd_data={25'd0,state==FAULT,lut_match,input_match,params_match,error_code!=0,done,busy};
      12'h010: rd_data=error_code;
      12'h014: rd_data=command;
      12'h018: rd_data=mode;
      12'h01c: rd_data=height;
      12'h020: rd_data=width;
      12'h024: rd_data=ci;
      12'h028: rd_data=co;
      12'h02c: rd_data=kernel;
      12'h030: rd_data=split;
      12'h034: rd_data=frame_id;
      12'h038: rd_data=job_id;
      12'h03c: rd_data=completed_frame;
      12'h040: rd_data=completed_job;
      12'h044: rd_data=cycles;
      12'h048: rd_data=expected_in;
      12'h04c: rd_data=expected_out;
      12'h050: rd_data=received_bytes;
      12'h054: rd_data=sent_bytes;
      12'h058: rd_data=compute_cycles;
      12'h05c: rd_data=relu_flags;
      default: rd_resp=2'b10;
    endcase
  end
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      command<=0; mode<=0; height<=0; width<=0; ci<=0; co<=0; kernel<=3;
      split<=0; frame_id<=0; job_id<=0; relu_flags<=0;
    end else if (wr_en && wr_resp==0) begin
      case(wr_addr)
        12'h014: command<=merged(command,wr_data,wr_strb);
        12'h018: mode<=merged(mode,wr_data,wr_strb);
        12'h01c: height<=merged(height,wr_data,wr_strb);
        12'h020: width<=merged(width,wr_data,wr_strb);
        12'h024: ci<=merged(ci,wr_data,wr_strb);
        12'h028: co<=merged(co,wr_data,wr_strb);
        12'h02c: kernel<=merged(kernel,wr_data,wr_strb);
        12'h030: split<=merged(split,wr_data,wr_strb);
        12'h05c: relu_flags<=merged(relu_flags,wr_data,wr_strb) & 32'd1;
        12'h034: frame_id<=merged(frame_id,wr_data,wr_strb);
        12'h038: job_id<=merged(job_id,wr_data,wr_strb);
        default: ;
      endcase
    end
  end

  assign load_ready=state==LOAD;
  assign s_axis_tready=unpack_ready && state==LOAD;
  axis_byte_unpack unpacker(.aclk,.aresetn(core_resetn),.s_axis_tdata,.s_axis_tkeep,
    .s_axis_tvalid(s_axis_tvalid && state==LOAD),.s_axis_tlast,.s_axis_tready(unpack_ready),
    .m_data(load_byte),.m_valid(load_valid),.m_last(load_last),.m_ready(load_ready),.protocol_error(unpack_error));
  assign write_byte=load_valid && load_ready && load_last==expected_last_input &&
    !((command==2 || command==3 || (command==1 && param_phase==0)) && load_byte==8'h80) &&
    !(command==1 && param_phase==2 && param_byte==3 && load_byte[7]);
  assign input_we=write_byte && command==2;
  // Register the checked/mapped byte before the banked BRAM write port.
  // LOAD_DRAIN keeps completion behind the final physical write.
  always_ff @(posedge aclk) begin
    if (!core_resetn) input_we_q<=0;
    else begin
      input_we_q<=input_we;
      if (input_we) begin
        input_bank_q<=input_channel[1:0]; input_address_q<=input_address;
        input_byte_q<=convolution_input_byte;
      end
    end
  end
  assign weight_we=write_byte && command==1 && param_phase==0;
  assign param_we=write_byte && command==1 && param_phase!=0 && param_byte==3;
  assign lut_we=write_byte && command==3;
  assign weight_address=weight_group_base+weight_input_base+weight_k;
  assign input_address=input_group_base+input_pixel;
  always_comb begin
    param_next=param_word;
    param_next[param_byte*8+:8]=load_byte;
  end
  assign output_address=output_group_base+output_pixel;
  // Reserve a queue slot for the synchronous RAM response already in flight.
  // The packer can stall at any beat; no read may overwrite a pending byte.
  assign output_pop=state==OUT_STREAM && output_queued!=0 && pack_ready;
  assign output_push=state==OUT_STREAM && output_read_pending && output_valid;
  assign output_read_enable=state==OUT_STREAM && output_issued<expected_out &&
    (output_queued+{2'd0,output_read_pending}<3'd4 || output_pop);
  assign result_byte=output_queue[output_rptr];
  conv_tile_engine_acdc conv(.aclk,.aresetn(core_resetn),.start(state==CORE_START && convolution_mode),
    .cfg_in_h(height[15:0]),.cfg_in_w(width[15:0]),.cfg_ci(ci[7:0]),.cfg_co(co[6:0]),.cfg_kernel3(kernel==3),
    .cfg_relu(relu_flags[0]),.cfg_fused_concat(1'b0),.cfg_split(split[7:0]),
    .busy(conv_busy),.done(conv_done),.error(conv_error),
    .input_we(input_we_q),.input_bank(input_bank_q),.input_addr(input_address_q),.input_data(input_byte_q),
    .weight_we,.weight_bank({weight_o[2:0],weight_i[1:0]}),.weight_addr(weight_address),.weight_data(load_byte),
    .param_we,.param_multiplier(param_phase==2),.param_channel,.param_data(param_next),
    .output_re(output_read_enable),.output_bank(output_channel[2:0]),.output_addr(output_address),.output_data,.output_valid,
    .aux_input_re,.aux_input_addr,.aux_input_data,.aux_input_valid,
    .aux_output_we,.aux_output_bank,.aux_output_addr,.aux_output_data);
  tensor_tile_engine_acdc tensor(.aclk,.aresetn(core_resetn),.start(state==CORE_START && !convolution_mode),
    .cfg_op(mode[1:0]-2'd1),.cfg_h(height[15:0]),.cfg_w(width[15:0]),.cfg_channels(ci[7:0]),.cfg_split(split[7:0]),
    .busy(tensor_busy),.done(tensor_done),.error(tensor_error),
    .lut_we,.lut_addr(received_bytes[8:0]),.lut_data(load_byte),
    .input_re(aux_input_re),.input_addr(aux_input_addr),.input_data(aux_input_data),.input_valid(aux_input_valid),
    .output_we(aux_output_we),.output_bank(aux_output_bank),.output_addr(aux_output_addr),.output_data(aux_output_data));
  axis_byte_pack packer(.aclk,.aresetn(core_resetn),.s_data(result_byte),.s_valid(state==OUT_STREAM && output_queued!=0),
    .s_last(output_count+1==expected_out),.s_ready(pack_ready),
    .m_axis_tdata,.m_axis_tkeep,.m_axis_tvalid,.m_axis_tlast,.m_axis_tready);

  task automatic fail_command(input logic [31:0] code);
    error_code<=code; done<=1; state<=FAULT;
    params_ready<=0; input_ready<=0; lut_ready<=0;
    completed_frame<=frame_id; completed_job<=job_id;
  endtask
  task automatic finish_command;
    done<=1; state<=IDLE;
    completed_frame<=frame_id; completed_job<=job_id;
  endtask
  always_ff @(posedge aclk) begin
    if (!aresetn || soft_reset_q) begin
      state<=IDLE; done<=0; error_code<=0;
      params_ready<=0; input_ready<=0; lut_ready<=0; lut_dual<=0;
      completed_frame<=0; completed_job<=0; cycles<=0; compute_cycles<=0;
      expected_in<=0; expected_out<=0; received_bytes<=0; sent_bytes<=0; output_count<=0;
      output_read_pending<=0; output_issued<=0; output_queued<=0; output_rptr<=0; output_wptr<=0;
    end else begin
      output_read_pending<=output_read_enable;
      if (busy) cycles<=cycles+1'b1;
      if (state==CORE_WAIT) compute_cycles<=compute_cycles+1'b1;
      if (m_axis_tvalid && m_axis_tready) sent_bytes<=sent_bytes+keep_count(m_axis_tkeep);
      case(state)
        IDLE: if (start_command) begin
          done<=0; error_code<=0; cycles<=0; compute_cycles<=0;
          expected_in<=0; expected_out<=0; received_bytes<=0; sent_bytes<=0; output_count<=0;
          // Stage the shape checks: only registers set through this START are
          // compared here, so the wide decode ends at one flip-flop instead
          // of every load-address enable in CHECK.
          check_shape<=command>=1 && command<=4 && mode<=4 && height>0 && width>0 &&
                       ci>0 && ci<=192 &&
                       ((mode==0 || mode==5) ?
                         (co>0 && co<=64 && (kernel==1 || kernel==3) &&
                          height>=kernel && width>=kernel && height<=258 && width<=258) :
                         mode==6 ?
                         (co>0 && co<=64 && kernel==3 && height>=3 && width>=3 &&
                          height<=258 && width<=258 && !height[0] && !width[0] &&
                          split>0 && split<ci && split[1:0]==0 && ci[1:0]==split[1:0]) :
                         (height<=256 && width<=256)) &&
                       !((mode==2 || mode==4) && (height[0] || width[0])) &&
                       !(mode==4 && !(split>0 && split<ci)) &&
                       !(command==1 && !(mode==0 || mode==5 || mode==6)) &&
                       !(command==3 && mode==0);
          if (command==1) params_ready<=0;
          if (command==2) input_ready<=0;
          if (command==3) begin
            lut_ready<=0;
            // A mapped input belongs to the LUT used during its load.
            if (tag_mode==5 || tag_mode==6) input_ready<=0;
          end
          state<=SHAPE1;
        end
        SHAPE1: begin
          plane<={23'd0,height[8:0]}*{23'd0,width[8:0]};
          deep_plane6<=(({23'd0,height[8:0]}>>1)+32'd1)*(({23'd0,width[8:0]}>>1)+32'd1);
          channel_product<={24'd0,ci[7:0]}*{25'd0,co[6:0]};
          ci_groups<=(ci[7:0]+8'd3)>>2;
          co_groups<=convolution_mode ? (co[6:0]+8'd7)>>3 : (ci[7:0]+8'd7)>>3;
          result_channels<=convolution_mode ? co[6:0] : ci[7:0];
          skip_groups<=(split[7:0]+8'd3)>>2;
          deep_groups<=({1'b0,ci[7:0]}-{1'b0,split[7:0]}+8'd3)>>2;
          deep_channels<=ci[7:0]-split[7:0];
          kernel_square<=kernel==3 ? 9 : 1;
          out_h<=convolution_mode ? height-kernel+1 : mode==2 ? height>>1 : mode==3 ? height<<1 : height;
          out_w<=convolution_mode ? width-kernel+1 : mode==2 ? width>>1 : mode==3 ? width<<1 : width;
          state<=SHAPE2;
        end
        SHAPE2: begin
          out_plane<={23'd0,out_h[8:0]}*{23'd0,out_w[8:0]};
          weight_bytes<=channel_product*kernel_square;
          weights_per_group<=ci_groups*kernel_square;
          deep_base<=skip_groups*plane;
          normal_input_bytes<=ci[7:0]*plane;
          skip_input_bytes<=split[7:0]*plane;
          deep_input_bytes<=deep_channels*(mode==6 ? deep_plane6 : plane>>2);
          deep_group_bytes<=deep_groups*(mode==6 ? deep_plane6 : plane>>2);
          check_dims<=check_shape && out_h>0 && out_w>0 && out_h<=256 && out_w<=256;
          state<=SHAPE3;
        end
        SHAPE3: begin
          weight_occupancy<=co_groups*weights_per_group;
          input_occupancy<=(mode==4 || mode==6) ? deep_base+deep_group_bytes : ci_groups*plane;
          output_occupancy<=co_groups*out_plane;
          expected_in<=command==1 ? weight_bytes+(co[6:0]<<3) : command==2 ? ((mode==4 || mode==6) ? skip_input_bytes+deep_input_bytes : normal_input_bytes) : command==3 ? ((mode==4 || mode==6) ? 512 : 256) : 0;
          expected_out<=command==4 ? result_channels*out_plane : 0;
          state<=CHECK;
        end
        CHECK: begin
          if (!check_dims || input_occupancy>32768 || output_occupancy>8192 || (convolution_mode && weight_occupancy>2048) || (mode!=0 && relu_flags[0])) fail_command(1);
          else if (command==2 && (mode==5 || mode==6) && !lut_match) fail_command(4);
          else if (command==4) begin
            if (!input_match || (convolution_mode && !params_match) || (mode!=0 && !lut_match)) fail_command(4);
            else state<=CORE_START;
          end else begin
            param_phase<=0; param_byte<=0; param_channel<=0; param_word<=0;
            weight_o<=0; weight_i<=0; weight_k<=0; weight_group_base<=0; weight_input_base<=0;
            input_channel<=0; input_pixel<=0; input_group_base<=0; input_branch<=0;
            input_plane<=plane[15:0]; input_channel_limit<=(mode==4 || mode==6) ? split[7:0] : ci[7:0];
            input_remaining<=expected_in; expected_last_input<=expected_in==1;
            state<=LOAD;
          end
        end
        LOAD: begin
          if (unpack_error) fail_command(3);
          else if (load_valid) begin
            received_bytes<=received_bytes+1'b1;
            input_remaining<=input_remaining-1'b1;
            expected_last_input<=input_remaining==2;
            if (load_last!=expected_last_input) fail_command(2);
            else if (!write_byte) fail_command(7);
            else begin
              if (command==1) begin
                if (param_phase==0) begin
                  if (weight_k!=kernel_square-1) weight_k<=weight_k+1'b1;
                  else begin
                    weight_k<=0;
                    if (weight_i!=ci[7:0]-1) begin
                      weight_i<=weight_i+1'b1;
                      if (weight_i[1:0]==3) weight_input_base<=weight_input_base+kernel_square;
                    end else begin
                      weight_i<=0; weight_input_base<=0;
                      if (weight_o!=co[6:0]-1) begin
                        weight_o<=weight_o+1'b1;
                        if (weight_o[2:0]==7) weight_group_base<=weight_group_base+weights_per_group;
                      end else param_phase<=1;
                    end
                  end
                end else begin
                  param_word<=param_next;
                  param_byte<=param_byte+1'b1;
                  if (param_byte==3) begin
                    if (param_channel!=co[6:0]-1) param_channel<=param_channel+1'b1;
                    else begin param_channel<=0; param_phase<=2; end
                  end
                end
              end
              if (command==2) begin
                if (input_pixel!=input_plane-1) input_pixel<=input_pixel+1'b1;
                else begin
                  input_pixel<=0;
                  if ((mode==4 || mode==6) && !input_branch && input_channel==input_channel_limit-1) begin
                    input_branch<=1; input_channel<=0; input_group_base<=deep_base[14:0];
                    input_plane<=mode==6 ? deep_plane6[15:0] : plane[15:0]>>2;
                    input_channel_limit<=ci[7:0]-split[7:0];
                  end else begin
                    input_channel<=input_channel+1'b1;
                    if (input_channel[1:0]==3) input_group_base<=input_group_base+input_plane;
                  end
                end
              end
              if (load_last) begin
                if (command==1) begin params_ready<=1; tag_ci<=ci; tag_co<=co; tag_kernel<=kernel; end
                if (command==2) begin input_ready<=1; tag_mode<=mode; tag_h<=height; tag_w<=width; tag_input_ci<=ci; tag_split<=split; end
                if (command==3) begin lut_ready<=1; lut_dual<=(mode==4 || mode==6); end
                state<=LOAD_DRAIN;
              end
            end
          end
        end
        LOAD_DRAIN: finish_command();
        CORE_START: state<=CORE_WAIT;
        CORE_WAIT: if (convolution_mode ? conv_done : tensor_done) begin
          if (convolution_mode ? conv_error!=0 : tensor_error!=0) fail_command(convolution_mode && conv_error==2 ? 5 : 6);
          else begin
            output_channel<=0; output_pixel<=0; output_group_base<=0;
            output_issued<=0; output_queued<=0; output_rptr<=0; output_wptr<=0;
            state<=OUT_STREAM;
          end
        end
        OUT_STREAM: begin
          case ({output_push,output_pop})
            2'b10: output_queued<=output_queued+1'b1;
            2'b01: output_queued<=output_queued-1'b1;
            default: ;
          endcase
          if (output_push) begin
            output_queue[output_wptr]<=output_data; output_wptr<=output_wptr+1'b1;
          end
          if (output_pop) begin
            output_rptr<=output_rptr+1'b1; output_count<=output_count+1'b1;
            if (output_count+1==expected_out) state<=OUT_DRAIN;
          end
          if (output_read_enable) begin
            output_issued<=output_issued+1'b1;
            if (output_pixel!=out_plane-1) output_pixel<=output_pixel+1'b1;
            else begin
              output_pixel<=0; output_channel<=output_channel+1'b1;
              if (output_channel[2:0]==7) output_group_base<=output_group_base+out_plane;
            end
          end
          if (output_read_pending && !output_valid) fail_command(6);
        end
        OUT_DRAIN: if (m_axis_tvalid && m_axis_tready && m_axis_tlast) finish_command();
        FAULT: ;
        default: fail_command(6);
      endcase
    end
  end
endmodule
