// One input branch into a parity page: identity, maxpool2 or nearest2.
// Concat is two descriptors: clear first, append second with the same epoch.
// Epoch must change whenever the source tensor contents change.
module tensor_load #(parameter integer AW=12) (
  input logic clk,resetn,start,
  input logic [31:0] cfg_source,cfg_epoch,
  input logic [8:0] cfg_width,cfg_height,cfg_y,cfg_rows,
  input logic [4:0] cfg_groups,cfg_branch_groups,cfg_group_offset,
  input logic [1:0] cfg_mode,
  input logic cfg_pad,cfg_page,cfg_table,cfg_clear,
  output logic busy,done,error,
  input logic lut_table,
  input logic [7:0] lut_index,lut_value,
  input logic lut_valid,
  output logic lut_ready,
  output logic [31:0] rd_addr,
  output logic [15:0] rd_words,
  output logic rd_cmd_valid,
  input logic rd_cmd_ready,
  input logic [63:0] rd_data,
  input logic rd_valid,rd_last,rd_done,rd_error,
  output logic rd_ready,
  output logic f_page,f_bank,
  output logic [AW-1:0] f_addr,
  output logic [63:0] f_data,
  output logic f_valid,
  input logic f_ready
);
  typedef enum logic [4:0] {IDLE,P1,P2,P3,CHECK,CLEAR,ROW,MUL,ADDR,OP,SEND,LOAD,NEXT,ABORT} state_t;
  state_t state;
  logic [8:0] width,height,y0,rows,first_y,last_y,yy,halfwidth;
  logic [9:0] sw,sh,tile_height,sy;
  logic [4:0] groups,branch_groups,offset,cg;
  logic [1:0] mode,op_mode;
  logic pad,page,table_sel,clear_page;
  logic [31:0] source_base,epoch,area,plane_pixels,source_words,requested_words;
  logic [31:0] row_stride,row_offset,src_offset,group_offset;
  logic [32:0] source_end;
  logic [AW:0] clear_addr,total_words;
  logic clear_bank,write_bank;
  logic [AW-1:0] write_addr;
  logic [9:0] out_count;
  logic [15:0] input_count;
  logic [63:0] mapped;
  logic op_busy,op_done,op_error,op_ready,op_valid,op_last,lut_ready_i;
  logic [1:0] page_valid;
  logic [73:0] page_signature0,page_signature1;
  logic [73:0] signature;
  // 32 epoch + four 9-bit dimensions + 5 groups + pad = 74.
  assign signature={epoch,width,height,y0,rows,groups,pad};
  assign busy=state!=IDLE;
  assign rd_cmd_valid=state==SEND;
  assign rd_ready=state==ABORT || (state==LOAD && op_ready);
  assign f_page=page;
  assign f_bank=state==CLEAR ? clear_bank : write_bank;
  assign f_addr=state==CLEAR ? clear_addr[AW-1:0] : write_addr;
  assign f_data=state==CLEAR ? 64'b0 : mapped;
  assign f_valid=resetn && (state==CLEAR || (state==LOAD && op_valid));
  assign lut_ready=!busy && lut_ready_i;
  row_transform transform(.clk,.resetn(resetn && state!=ABORT),.start(state==OP),
    .cfg_mode(op_mode),.cfg_width(sw),.cfg_table(table_sel),.busy(op_busy),.done(op_done),.error(op_error),
    .lut_table,.lut_index,.lut_value,.lut_valid(lut_valid && !busy),.lut_ready(lut_ready_i),
    .s_data(rd_data),.s_valid(rd_valid && state==LOAD),.s_ready(op_ready),
    .m_data(mapped),.m_valid(op_valid),.m_last(op_last),.m_ready(state==LOAD && f_ready));
  always_ff @(posedge clk) begin
    if(!resetn) begin state<=IDLE;done<=0;error<=0;page_valid<=0;end
    else begin
      done<=0;
      if(lut_valid && lut_ready) page_valid<=0;
      case(state)
        IDLE: if(start) begin
          error<=0;
          if(cfg_source[2:0]!=0 || cfg_width==0 || cfg_width>256 || cfg_height==0 || cfg_height>256 ||
             cfg_rows==0 || {1'b0,cfg_y}+cfg_rows>cfg_height || cfg_groups==0 || cfg_groups>24 ||
             cfg_branch_groups==0 || {1'b0,cfg_group_offset}+cfg_branch_groups>cfg_groups || cfg_mode>2 ||
             (cfg_mode==2 && (cfg_width[0] || cfg_height[0]))) begin error<=1;done<=1;end
          else begin
            width<=cfg_width;height<=cfg_height;y0<=cfg_y;rows<=cfg_rows;groups<=cfg_groups;
            branch_groups<=cfg_branch_groups;offset<=cfg_group_offset;mode<=cfg_mode;
            pad<=cfg_pad;page<=cfg_page;table_sel<=cfg_table;clear_page<=cfg_clear;
            epoch<=cfg_epoch;source_base<=cfg_source;
            sw<=cfg_mode==1 ? {cfg_width,1'b0} : cfg_mode==2 ? cfg_width>>1 : cfg_width;
            sh<=cfg_mode==1 ? {cfg_height,1'b0} : cfg_mode==2 ? cfg_height>>1 : cfg_height;
            first_y<=cfg_pad && cfg_y!=0 ? cfg_y-1 : cfg_y;
            yy<=cfg_pad && cfg_y!=0 ? cfg_y-1 : cfg_y;
            last_y<=({1'b0,cfg_y}+cfg_rows+cfg_pad>cfg_height) ? cfg_height : cfg_y+cfg_rows+cfg_pad;
            halfwidth<=({1'b0,cfg_width}+2*cfg_pad+1)>>1;tile_height<={1'b0,cfg_rows}+2*cfg_pad;
            clear_addr<=0;clear_bank<=0;cg<=0;state<=P1;
          end
        end
        P1: begin area<=halfwidth*tile_height;plane_pixels<=sw*sh;row_stride<=halfwidth*groups;state<=P2;end
        P2: begin requested_words<=area*groups;source_words<=plane_pixels*branch_groups;state<=P3;end
        P3: begin source_end<={1'b0,source_base}+({1'b0,source_words}<<3);state<=CHECK;end
        CHECK: begin
          if(requested_words>(1<<AW) || source_end>33'h100000000 ||
             (!clear_page && (!page_valid[page] || (page ? page_signature1 : page_signature0)!=signature))) begin
            error<=1;done<=1;state<=IDLE;
          end else begin
            total_words<=requested_words;page_valid[page]<=0;state<=clear_page ? CLEAR : ROW;
          end
        end
        CLEAR: if(f_ready) begin
          clear_bank<=!clear_bank;
          if(clear_bank) begin clear_addr<=clear_addr+1;if(clear_addr==total_words-1) state<=ROW;end
        end
        ROW: begin
          sy<=mode==1 ? {yy,1'b0} : mode==2 ? yy>>1 : yy;
          op_mode<=mode==2 ? ((yy!=first_y && yy[0]) ? 3 : 2) : mode;
          row_offset<=(yy+pad-y0)*row_stride;group_offset<=cg*plane_pixels;state<=MUL;
        end
        MUL: begin src_offset<=sy*sw;state<=ADDR;end
        ADDR: begin
          rd_addr<=source_base+((group_offset+src_offset)<<3);
          rd_words<=mode==1 ? sw<<1 : sw;
          write_addr<=row_offset+offset+cg;write_bank<=pad;out_count<=0;input_count<=0;state<=OP;
        end
        OP: state<=op_mode==3 ? LOAD : SEND;
        SEND: if(rd_cmd_ready) state<=LOAD;
        LOAD: begin
          if(rd_valid && rd_ready) begin
            input_count<=input_count+1;
            if(rd_last!==(input_count==rd_words-1)) begin error<=1;page_valid[page]<=0;state<=ABORT;end
          end
          if(f_valid && f_ready) begin
            out_count<=out_count+1;write_bank<=!write_bank;
            if(write_bank) write_addr<=write_addr+groups;
            if(op_last) begin
              if(out_count!=width-1) begin error<=1;state<=ABORT;end
              else state<=NEXT;
            end
          end
          if(rd_error || op_error) begin error<=1;page_valid[page]<=0;state<=ABORT;end
        end
        NEXT: begin
          if(yy+1<last_y) begin yy<=yy+1;state<=ROW;end
          else if(cg+1<branch_groups) begin cg<=cg+1;yy<=first_y;state<=ROW;end
          else begin
            if(page) page_signature1<=signature;else page_signature0<=signature;
            page_valid[page]<=1;done<=1;state<=IDLE;
          end
        end
        ABORT: if(rd_cmd_ready) begin page_valid[page]<=0;done<=1;state<=IDLE;end
        default: begin state<=IDLE;error<=1;page_valid<=0;end
      endcase
    end
  end
endmodule
