// Full-image C8HW8 DDR -> halo tile banks. CAT reads skip/deep separately,
// applies exact branch LUTs, caches one source row and expands nearest x2.
// All dimensions are powers of two (the frozen network's 32/64/128/256 grids).
module input_loader #(
  parameter integer P=4,
  parameter integer WORDS=(P==4?4608:3584),
  parameter integer AW=$clog2(WORDS),
  parameter integer TIMEOUT_CYCLES=10000000
) (
  input logic aclk,aresetn,start,
  input logic [31:0] cfg_source0,cfg_source1,
  input logic [15:0] cfg_image_h,cfg_image_w,cfg_y,cfg_x,cfg_oh,cfg_ow,
  input logic [7:0] cfg_ci,cfg_split,
  input logic cfg_kernel3,cfg_cat,cfg_slot,
  input logic lut_we,
  input logic [8:0] lut_addr,
  input logic [7:0] lut_data,
  output logic busy,done,
  output logic [7:0] error,
  output logic [71:0] cmd_data,
  output logic cmd_valid,
  input logic cmd_ready,
  input logic [7:0] status_data,
  input logic status_valid,
  output logic status_ready,
  input logic [63:0] dma_data,
  input logic [7:0] dma_keep,
  input logic dma_last,dma_valid,
  output logic dma_ready,
  output logic load_slot,
  output logic [$clog2(P)-1:0] load_bank,
  output logic [AW-1:0] load_addr,
  output logic [63:0] load_data,
  output logic [7:0] load_keep,
  output logic load_valid,
  input logic load_ready
);
  typedef enum logic [3:0] {IDLE,INITIALIZE,CHECK_PLANE,CHECK_CAPACITY,PREPARE_ROW,ADDRESS_ROW,REQUEST_ROW,READ_ROW,EXPAND,ROW_DRAIN,NEXT_ROW,FAULT} state_t;
  state_t state;
  logic [31:0] source1,group_address,plane_bytes,request_address;
  logic [15:0] image_h,image_w,tile_y,tile_x,oh,ow,ih,iw,columns,row,bank_row_base;
  logic [8:0] column_numerator,column_count;
  logic [7:0] ci,split;
  logic [4:0] cg,cg_count,split_groups;
  logic pad,cat,slot,deep,cache_valid,row_zero;
  logic [3:0] log_h,log_w;
  logic [7:0] cache_y,source_y;
  logic [15:0] request_x;
  logic [8:0] request_words,received_words,rx_index,col,word_column;
  logic [$clog2(P)-1:0] bank_counter;
  logic signed [17:0] gy,gx,left_x,right_x,clipped_left,clipped_right;
  logic zero_pixel,producer_ready,producer_issue,out_zero;
  logic [7:0] cache_index;
  logic [63:0] cache_read;
  (* ram_style="block" *) logic [63:0] row_memory[256];
  logic [63:0] raw_data,tf_data;
  logic [7:0] raw_keep,lane_mask;
  logic raw_last,raw_valid,raw_ready,tf_ready,tf_valid,tf_invalid;
  logic [31:0] tf_tag;
  logic request_ready,transfer_busy,transfer_done,transfer_fault,finished;
  logic [7:0] transfer_error;
  logic [31:0] actual_bytes;
  (* use_dsp="no" *) logic [31:0] capacity;
  (* use_dsp="no" *) logic [15:0] plane_words;
  function automatic logic [3:0] log_dimension(input logic [15:0] value);
    log_dimension=15;
    for(int i=0;i<=8;i++)if(value==(1<<i))log_dimension=4'(i);
  endfunction
  assign busy=state!=IDLE;
  assign load_slot=slot;assign load_keep=8'hff;
  assign load_data=out_zero?64'd0:cache_read;
  assign producer_ready=!load_valid||load_ready;
  assign producer_issue=state==EXPAND&&producer_ready;
  assign deep=cat&&cg>=split_groups;
  // Width is validated before INITIALIZE; halo width is at most 258.
  assign column_numerator=iw[8:0]+9'(P-1);
  assign column_count=column_numerator/9'(P);
  always_comb begin
    gy=$signed({2'd0,tile_y})+$signed({2'd0,row})-$signed({17'd0,pad});
    gx=$signed({2'd0,tile_x})+$signed({9'd0,col})-$signed({17'd0,pad});
    left_x=$signed({2'd0,tile_x})-$signed({17'd0,pad});
    right_x=$signed({2'd0,tile_x})+$signed({2'd0,ow})+$signed({17'd0,pad});
    clipped_left=left_x<0?18'sd0:left_x;
    clipped_right=right_x>image_w?$signed({2'd0,image_w}):right_x;
    row_zero=gy<0||gy>=image_h;
    zero_pixel=row_zero||gx<0||gx>=image_w;
    source_y=deep?8'(gy>>>1):8'(gy);
    cache_index=zero_pixel?0:8'((deep?(gx>>>1):gx)-$signed({2'd0,request_x}));
    for(int i=0;i<8;i++)lane_mask[i]=(cg*8+i)<ci;
  end
  assign raw_ready=state==READ_ROW&&tf_ready;
  input_transform transform(.aclk,.aresetn,.lut_we(lut_we&&!busy),.lut_addr,.lut_data,
    .s_data(raw_data),.s_lane_mask(lane_mask),.s_tag({23'd0,rx_index}),
    .s_lookup(cat),.s_branch(deep),.s_zero(1'b0),.s_valid(raw_valid&&state==READ_ROW),.s_ready(tf_ready),
    .m_data(tf_data),.m_tag(tf_tag),.m_valid(tf_valid),.m_ready(1'b1),.invalid_int8(tf_invalid));
  dma_transfer #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) transfer(.aclk,.aresetn,
    .req_address(request_address),.req_bytes({11'd0,request_words,3'd0}),.req_tag(4'h2),
    .req_valid(state==REQUEST_ROW),.req_ready(request_ready),.cmd_data,.cmd_valid,.cmd_ready,
    .status_data,.status_valid,.status_ready,.s_data(dma_data),.s_keep(dma_keep),.s_last(dma_last),
    .s_valid(dma_valid),.s_ready(dma_ready),.m_data(raw_data),.m_keep(raw_keep),.m_last(raw_last),
    .m_valid(raw_valid),.m_ready(raw_ready),.busy(transfer_busy),.done(transfer_done),
    .fault(transfer_fault),.error(transfer_error),.actual_bytes);
  always_ff @(posedge aclk)begin
    if(tf_valid&&state==READ_ROW&&tf_tag<256)row_memory[tf_tag[7:0]]<=tf_data;
    if(producer_issue&&!zero_pixel)cache_read<=row_memory[cache_index];
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)begin state<=IDLE;done<=0;error<=0;load_valid<=0;cache_valid<=0;end
    else begin
      done<=0;
      if(load_valid&&load_ready)load_valid<=0;
      if(transfer_done)finished<=1;
      if(raw_valid&&raw_ready)rx_index<=rx_index+1;
      if(tf_valid&&state==READ_ROW)received_words<=received_words+1;
      case(state)
        IDLE:if(start)begin
          error<=0;source1<=cfg_source1;group_address<=cfg_source0;
          image_h<=cfg_image_h;image_w<=cfg_image_w;tile_y<=cfg_y;tile_x<=cfg_x;
          oh<=cfg_oh;ow<=cfg_ow;ih<=cfg_oh+(cfg_kernel3?2:0);iw<=cfg_ow+(cfg_kernel3?2:0);
          ci<=cfg_ci;split<=cfg_split;cg_count<=({1'b0,cfg_ci}+7)>>3;split_groups<=cfg_split>>3;
          pad<=cfg_kernel3;cat<=cfg_cat;slot<=cfg_slot;
          log_h<=log_dimension(cfg_image_h);log_w<=log_dimension(cfg_image_w);
          cg<=0;row<=0;bank_row_base<=0;cache_valid<=0;load_valid<=0;
          if(log_dimension(cfg_image_h)==15||log_dimension(cfg_image_w)==15||
             cfg_ci==0||cfg_ci>192||cfg_oh==0||cfg_ow==0||cfg_oh>256||cfg_ow>256||
             {1'b0,cfg_y}+cfg_oh>cfg_image_h||{1'b0,cfg_x}+cfg_ow>cfg_image_w||
             cfg_source0[2:0]!=0||
             (cfg_cat&&(cfg_source1[2:0]!=0||cfg_split==0||cfg_split>=cfg_ci||cfg_split[2:0]!=0||cfg_image_h<2||cfg_image_w<2)))begin
            state<=FAULT;error<=1;done<=1;
          end else state<=INITIALIZE;
        end
        INITIALIZE:begin columns<={7'd0,column_count};plane_bytes<=32'd1<<(log_h+log_w+3);state<=CHECK_PLANE;end
        CHECK_PLANE:begin plane_words<=ih*columns;state<=CHECK_CAPACITY;end
        CHECK_CAPACITY:begin capacity<=plane_words*cg_count;state<=PREPARE_ROW;end
        PREPARE_ROW:begin
          if(capacity>WORDS)begin state<=FAULT;error<=1;done<=1;end
          else begin
            request_x<=deep?16'(clipped_left>>>1):16'(clipped_left);
            request_words<=deep?9'(((clipped_right-1)>>>1)-(clipped_left>>>1)+1):9'(clipped_right-clipped_left);
            col<=0;bank_counter<=0;word_column<=0;
            if(row_zero||(cache_valid&&cache_y==source_y))state<=EXPAND;
            else state<=ADDRESS_ROW;
          end
        end
        ADDRESS_ROW:begin
          request_address<=group_address+({24'd0,source_y}<<(log_w+3-(deep?1:0)))+({16'd0,request_x}<<3);
          state<=REQUEST_ROW;finished<=0;rx_index<=0;received_words<=0;
        end
        REQUEST_ROW:if(request_ready)state<=READ_ROW;
        READ_ROW:if(finished&&received_words==request_words)begin cache_valid<=1;cache_y<=source_y;state<=EXPAND;end
        EXPAND:if(producer_ready)begin
          load_valid<=1;load_bank<=bank_counter;load_addr<=AW'(bank_row_base+word_column);out_zero<=zero_pixel;
          if(col+1==iw)state<=ROW_DRAIN;
          else begin
            col<=col+1;
            if(bank_counter==P-1)begin bank_counter<=0;word_column<=word_column+1;end
            else bank_counter<=bank_counter+1;
          end
        end
        ROW_DRAIN:if(load_valid&&load_ready)state<=NEXT_ROW;
        NEXT_ROW:begin
          bank_row_base<=bank_row_base+columns;
          if(row+1<ih)begin row<=row+1;state<=PREPARE_ROW;end
          else if(cg+1<cg_count)begin
            row<=0;cg<=cg+1;cache_valid<=0;state<=PREPARE_ROW;
            if(cat&&cg+1==split_groups)group_address<=source1;
            else group_address<=group_address+(deep?(plane_bytes>>2):plane_bytes);
          end else begin state<=IDLE;done<=1;end
        end
        FAULT:;
        default:state<=FAULT;
      endcase
      if(state!=IDLE&&state!=FAULT&&(transfer_fault||tf_invalid))begin
        state<=FAULT;error<=transfer_fault?(8'h80|transfer_error):8'h04;done<=1;
      end
      if(lut_we&&busy)begin state<=FAULT;error<=8'h08;done<=1;end
    end
  end
endmodule
