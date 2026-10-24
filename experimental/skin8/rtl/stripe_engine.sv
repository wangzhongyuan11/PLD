// One autonomous transformed-input stripe/output block: DDR weights -> feature load
// -> 128-MAC convolution -> INT8 quantization -> DDR writeback.
// A higher-level graph sequencer will reuse features/weights between descriptors.
module stripe_engine (
  input logic clk,resetn,start,
  input logic [31:0] cfg_source,cfg_source2,cfg_weights,cfg_destination,
  input logic [1:0] cfg_mode,
  input logic cfg_concat,cfg_table2,
  input logic [4:0] cfg_branch_groups,
  input logic [8:0] cfg_width,cfg_height,cfg_y,cfg_rows,
  input logic [4:0] cfg_groups,
  input logic cfg_k3,cfg_relu,cfg_table,cfg_page,
  input logic cfg_reuse_features,
  input logic [31:0] cfg_feature_id,
  input logic [255:0] cfg_bias,cfg_multiplier,
  output logic busy,done,error,
  input logic lut_table,
  input logic [7:0] lut_index,lut_value,
  input logic lut_valid,
  output logic lut_ready,
  output logic [31:0] araddr,
  output logic [7:0] arlen,
  output logic [2:0] arsize,
  output logic [1:0] arburst,
  output logic arvalid,
  input logic arready,
  input logic [63:0] rdata,
  input logic [1:0] rresp,
  input logic rlast,rvalid,
  output logic rready,
  output logic [31:0] awaddr,
  output logic [7:0] awlen,
  output logic [2:0] awsize,
  output logic [1:0] awburst,
  output logic awvalid,
  input logic awready,
  output logic [63:0] wdata,
  output logic [7:0] wstrb,
  output logic wlast,wvalid,
  input logic wready,
  input logic [1:0] bresp,
  input logic bvalid,
  output logic bready
);
  typedef enum logic [3:0] {IDLE,PREP,WCMD,WLOAD,FSTART,FLOAD,OCMD,KSTART,RUN,DRAIN,FAIL} state_t;
  state_t state;
  logic [31:0] source,source2,weights,destination,feature_id;
  logic [1:0] mode;
  logic concat,table2,second_branch;
  logic [4:0] branch_groups;
  logic [8:0] width,height,y,rows;
  logic [4:0] groups;
  logic k3,relu,table_sel,page;
  logic reuse_features;
  logic [1:0] feature_valid;
  // Explicit page registers: do not infer a tiny asynchronous signature RAM.
  logic [147:0] feature_signature0,feature_signature1;
  logic [147:0] signature_q,requested_signature;
  logic [255:0] bias,multiplier;
  logic [7:0] weight_steps,weight_addr;
  logic [15:0] output_words;
  logic [2:0] weight_lane;
  logic rd_cmd_ready,rd_done,rd_error,rd_valid,rd_last,rd_ready;
  logic [63:0] rd_data;
  logic [31:0] load_addr;
  logic [15:0] load_words;
  logic load_cmd_valid,load_rd_ready,load_busy,load_done,load_error;
  logic loader_lut_ready;
  logic f_page,f_bank,f_valid,f_ready;
  logic [11:0] f_addr;
  logic [63:0] f_data;
  logic tile_busy,tile_done,tile_error,weight_ready;
  logic [127:0] tile_data;
  logic [15:0] tile_tag;
  logic tile_last,tile_second,tile_overflow,tile_valid,tile_ready;
  logic [63:0] words_data;
  logic words_last,words_valid,words_ready,writer_ready,writer_done,writer_error,writer_data_ready;
  assign busy=state!=IDLE;
  // feature_id is a sequencer epoch: a new frame/layer/stripe must reload first.
  // This caches input bytes across output blocks, never prior inference results.
  assign requested_signature={cfg_feature_id,cfg_source,cfg_width,cfg_height,cfg_y,cfg_rows,cfg_groups,cfg_k3,cfg_table,cfg_source2,cfg_mode,cfg_concat,cfg_table2,cfg_branch_groups};
  assign rd_ready=state==FAIL ? 1'b1 : state==WLOAD ? weight_ready : load_rd_ready;
  assign lut_ready=!busy && loader_lut_ready;
  assign words_ready=state==DRAIN ? 1'b1 : writer_data_ready;
  ddr_read64 reader(.clk,.resetn,
    .cmd_addr(state==WCMD ? weights : load_addr),.cmd_words(state==WCMD ? {5'b0,weight_steps,3'b0} : load_words),
    .cmd_valid(state==WCMD || load_cmd_valid),.cmd_ready(rd_cmd_ready),.done(rd_done),.error(rd_error),
    .araddr,.arlen,.arsize,.arburst,.arvalid,.arready,.rdata,.rresp,.rlast,.rvalid,.rready,
    .m_data(rd_data),.m_valid(rd_valid),.m_last(rd_last),.m_ready(rd_ready));
  tensor_load loader(.clk,.resetn,.start(state==FSTART && !rd_error && !reuse_features),.cfg_source(second_branch ? source2 : source),.cfg_width(width),.cfg_height(height),
    .cfg_y(y),.cfg_rows(rows),.cfg_groups(groups),.cfg_pad(k3),.cfg_page(page),.cfg_table(second_branch ? table2 : table_sel),
    .cfg_epoch(feature_id),.cfg_mode(second_branch ? 2'd2 : mode),.cfg_clear(!second_branch),
    .cfg_branch_groups(second_branch ? groups-branch_groups : branch_groups),.cfg_group_offset(second_branch ? branch_groups : 5'd0),
    .busy(load_busy),.done(load_done),.error(load_error),.lut_table,.lut_index,.lut_value,
    .lut_valid(lut_valid && !busy),.lut_ready(loader_lut_ready),.rd_addr(load_addr),.rd_words(load_words),.rd_cmd_valid(load_cmd_valid),
    .rd_cmd_ready,.rd_data,.rd_valid(rd_valid && state==FLOAD),.rd_last,.rd_done,.rd_error,.rd_ready(load_rd_ready),
    .f_page,.f_bank,.f_addr,.f_data,.f_valid,.f_ready);
  tile_int8 tile(.clk,.resetn,.start(state==KSTART && !writer_error),.cfg_page(page),.cfg_k3(k3),.cfg_relu(relu),
    .cfg_iw(width+(k3?9'd2:9'd0)),.cfg_ih(rows+(k3?9'd2:9'd0)),.cfg_ci_groups(groups),.cfg_bias(bias),.cfg_multiplier(multiplier),
    .busy(tile_busy),.done(tile_done),.error(tile_error),.f_page,.f_bank,.f_addr,.f_data,.f_valid,.f_ready,
    .w_lane(weight_lane),.w_addr(weight_addr),.w_data(rd_data),.w_valid(rd_valid && state==WLOAD),.w_ready(weight_ready),
    .m_data(tile_data),.m_tag(tile_tag),.m_last(tile_last),.m_second_valid(tile_second),.m_overflow(tile_overflow),.m_valid(tile_valid),.m_ready(tile_ready));
  pair_to_words serializer(.clk,.resetn,.s_data(tile_data),.s_second(tile_second),.s_last(tile_last),.s_valid(tile_valid),.s_ready(tile_ready),
    .m_data(words_data),.m_last(words_last),.m_valid(words_valid),.m_ready(words_ready));
  ddr_write64 writer(.clk,.resetn,.cmd_addr(destination),.cmd_words(output_words),.cmd_valid(state==OCMD),
    .cmd_ready(writer_ready),.done(writer_done),.error(writer_error),.s_data(words_data),.s_strb(8'hff),
    .s_valid(words_valid && state!=DRAIN),.s_ready(writer_data_ready),.awaddr,.awlen,.awsize,.awburst,.awvalid,.awready,
    .wdata,.wstrb,.wlast,.wvalid,.wready,.bresp,.bvalid,.bready);
  always_ff @(posedge clk) begin
    if(!resetn) begin state<=IDLE;done<=0;error<=0;feature_valid<=0;end
    else begin
      done<=0;
      if(lut_valid && lut_ready) feature_valid<=0;
      if(tile_valid && tile_ready && tile_overflow) error<=1;
      case(state)
        IDLE: if(start) begin
          error<=0;source<=cfg_source;source2<=cfg_source2;mode<=cfg_mode;concat<=cfg_concat;table2<=cfg_table2;
          branch_groups<=cfg_branch_groups;second_branch<=0;feature_id<=cfg_feature_id;weights<=cfg_weights;destination<=cfg_destination;
          width<=cfg_width;height<=cfg_height;y<=cfg_y;rows<=cfg_rows;groups<=cfg_groups;
          k3<=cfg_k3;relu<=cfg_relu;table_sel<=cfg_table;page<=cfg_page;bias<=cfg_bias;multiplier<=cfg_multiplier;
          reuse_features<=cfg_reuse_features;signature_q<=requested_signature;
          weight_steps<=cfg_groups*(cfg_k3?9:1);weight_addr<=0;weight_lane<=0;
          if(cfg_groups==0 || cfg_groups>24 || cfg_width==0 || cfg_width>256 || cfg_rows==0 || cfg_height==0 || cfg_height>256 ||
             {1'b0,cfg_y}+cfg_rows>cfg_height || cfg_mode>2 || cfg_branch_groups==0 ||
             (cfg_concat ? (cfg_branch_groups>=cfg_groups || cfg_mode!=0 || cfg_width[0] || cfg_height[0]) : cfg_branch_groups!=cfg_groups) ||
             (cfg_mode==2 && (cfg_width[0] || cfg_height[0])) ||
             cfg_source[2:0]!=0 || cfg_weights[2:0]!=0 || cfg_destination[2:0]!=0 || (cfg_concat && cfg_source2[2:0]!=0) ||
             (cfg_reuse_features && ((lut_valid && lut_ready) || !feature_valid[cfg_page] ||
              (cfg_page ? feature_signature1 : feature_signature0)!=requested_signature))) begin error<=1;done<=1;end
          else begin state<=PREP;if(!cfg_reuse_features) feature_valid[cfg_page]<=0;end
        end
        PREP: begin output_words<=width*rows;state<=WCMD;end
        WCMD: if(rd_cmd_ready) state<=WLOAD;
        WLOAD: begin
          if(rd_valid && rd_ready) begin
            if(weight_addr==weight_steps-1) begin weight_addr<=0;weight_lane<=weight_lane+1;end
            else weight_addr<=weight_addr+1;
            if(rd_last) state<=FSTART;
          end
          if(rd_error) begin error<=1;state<=FAIL;end
        end
        FSTART: if(rd_error) begin error<=1;state<=FAIL;end else state<=reuse_features ? OCMD : FLOAD;
        FLOAD: if(load_done) begin
          if(load_error) begin error<=1;state<=FAIL;end
          else if(concat && !second_branch) begin second_branch<=1;state<=FSTART;end
          else begin
            feature_valid[page]<=1;
            if(page) feature_signature1<=signature_q;else feature_signature0<=signature_q;
            state<=OCMD;
          end
        end
        OCMD: if(writer_ready) state<=KSTART;
        KSTART: if(writer_error) begin error<=1;state<=FAIL;end else state<=RUN;
        RUN: begin
          if(writer_done) begin
            if(writer_error) begin error<=1;state<=DRAIN;end
            else begin state<=IDLE;done<=1;end
          end
        end
        DRAIN: begin feature_valid<=0;if(!tile_busy && !words_valid) begin state<=IDLE;done<=1;end end
        FAIL: begin feature_valid<=0;if(rd_cmd_ready && !load_busy) begin state<=IDLE;done<=1;end end
        default: begin state<=FAIL;error<=1;end
      endcase
    end
  end
endmodule
