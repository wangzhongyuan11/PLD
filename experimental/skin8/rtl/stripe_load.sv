// NCHWc8 DDR rows -> mapped parity-banked feature page, with zero halo.
// Normal (non-pool/non-upsample) layers. One descriptor loads all Ci groups.
// Read-command/data ports connect to ddr_read64; LUT uses raw byte indices.
module stripe_load #(parameter integer AW=12) (
  input logic clk,resetn,start,
  input logic [31:0] cfg_source,
  input logic [8:0] cfg_width,cfg_height,cfg_y,cfg_rows,
  input logic [4:0] cfg_groups,
  input logic cfg_pad,cfg_page,cfg_table,
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
  typedef enum logic [3:0] {IDLE,PREP1,PREP2,PREP3,CHECK,CLEAR,ROW,ROW_MUL,ROW_ADDR,SEND,LOAD,ABORT} state_t;
  state_t state;
  logic [8:0] width,height,y_start,source_y,first_y,last_y,halfwidth;
  logic [4:0] groups,cg;
  logic pad,page,table_sel;
  logic [AW:0] total_words,clear_addr;
  logic clear_bank,write_bank,meta_bank,meta_last;
  logic [AW-1:0] write_addr,meta_addr;
  logic [31:0] row_address,first_address,row_bytes,plane_bytes;
  logic [9:0] x;
  logic map_valid,map_ready;
  logic [63:0] mapped;
  logic [31:0] requested_words;
  logic [32:0] source_end;
  logic [31:0] source_base,config_area,plane_pixels,source_words,first_offset,row_stride,row_offset;
  logic [9:0] tile_height;
  logic [8:0] row_y;
  logic [9:0] config_iw,config_ih;
  assign config_iw={1'b0,cfg_width}+(cfg_pad?2:0);
  assign config_ih={1'b0,cfg_rows}+(cfg_pad?2:0);
  assign busy=state!=IDLE;
  assign rd_addr=row_address;assign rd_words={7'b0,width};
  assign rd_cmd_valid=state==SEND;
  assign f_page=page;
  assign f_bank=state==CLEAR ? clear_bank : meta_bank;
  assign f_addr=state==CLEAR ? clear_addr[AW-1:0] : meta_addr;
  assign f_data=state==CLEAR ? 64'b0 : mapped;
  assign f_valid=resetn && (state==CLEAR || (state==LOAD && map_valid));
  logic map_input_ready;
  assign rd_ready=state==LOAD ? map_input_ready : state==ABORT;
  assign map_ready=state==LOAD ? f_ready : 1'b1;
  map8 mapper(.clk,.resetn,.cfg_lock(busy),.cfg_table(lut_table),.cfg_index(lut_index),.cfg_value(lut_value),
    .cfg_valid(lut_valid),.cfg_ready(lut_ready),.s_data(rd_data),.s_table(table_sel),
    .s_valid(rd_valid && state==LOAD),.s_ready(map_input_ready),.m_data(mapped),.m_valid(map_valid),.m_ready(map_ready));
  always_ff @(posedge clk) begin
    if(!resetn) begin state<=IDLE;done<=0;error<=0;end
    else begin
      done<=0;
      case(state)
        IDLE: if(start) begin
          error<=0;
          if(cfg_source[2:0]!=0 || cfg_width==0 || cfg_width>256 || cfg_height==0 || cfg_height>256 || cfg_rows==0 || cfg_groups==0 || cfg_groups>24 ||
             {1'b0,cfg_y}+cfg_rows>cfg_height) begin error<=1;done<=1;end
          else begin
            width<=cfg_width;height<=cfg_height;y_start<=cfg_y;groups<=cfg_groups;
            pad<=cfg_pad;page<=cfg_page;table_sel<=cfg_table;
            halfwidth<=(config_iw+1)>>1;tile_height<=config_ih;source_base<=cfg_source;
            first_y<=(cfg_pad && cfg_y!=0) ? cfg_y-1 : cfg_y;
            source_y<=(cfg_pad && cfg_y!=0) ? cfg_y-1 : cfg_y;
            last_y<=({1'b0,cfg_y}+cfg_rows+cfg_pad>cfg_height) ? cfg_height : cfg_y+cfg_rows+cfg_pad;
            row_bytes<={23'b0,cfg_width}<<3;
            clear_addr<=0;clear_bank<=0;cg<=0;state<=PREP1;
          end
        end
        PREP1: begin
          config_area<=halfwidth*tile_height;plane_pixels<=width*height;
          first_offset<=first_y*width;row_stride<=halfwidth*groups;state<=PREP2;
        end
        PREP2: begin
          requested_words<=config_area*groups;source_words<=plane_pixels*groups;
          first_address<=source_base+(first_offset<<3);row_address<=source_base+(first_offset<<3);state<=PREP3;
        end
        PREP3: begin
          source_end<={1'b0,source_base}+({1'b0,source_words}<<3);
          plane_bytes<=plane_pixels<<3;state<=CHECK;
        end
        CHECK: begin
          if(requested_words>(1<<AW) || source_end>33'h100000000) begin error<=1;done<=1;state<=IDLE;end
          else begin total_words<=requested_words;state<=CLEAR;end
        end
        CLEAR: if(f_ready) begin
          clear_bank<=!clear_bank;
          if(clear_bank) begin
            clear_addr<=clear_addr+1;
            if(clear_addr==total_words-1) state<=ROW;
          end
        end
        ROW: begin
          row_y<=source_y+pad-y_start;state<=ROW_MUL;
        end
        ROW_MUL: begin row_offset<=row_y*row_stride;state<=ROW_ADDR;end
        ROW_ADDR: begin
          write_addr<=row_offset+cg;
          write_bank<=pad;x<=0;state<=SEND;
        end
        SEND: if(rd_cmd_ready) state<=LOAD;
        LOAD: begin
          if(rd_valid && rd_ready) begin
            meta_addr<=write_addr;meta_bank<=write_bank;meta_last<=x==width-1;
            if(rd_last!==(x==width-1)) begin error<=1;state<=ABORT;end
            write_bank<=!write_bank;x<=x+1;
            if(write_bank) write_addr<=write_addr+groups;
          end
          if(f_valid && f_ready && meta_last) begin
            if(source_y+1<last_y) begin source_y<=source_y+1;row_address<=row_address+row_bytes;state<=ROW;end
            else if(cg+1<groups) begin
              cg<=cg+1;source_y<=first_y;first_address<=first_address+plane_bytes;row_address<=first_address+plane_bytes;state<=ROW;
            end else begin state<=IDLE;done<=1;end
          end
          if(rd_error) begin error<=1;state<=ABORT;end
        end
        ABORT: if(rd_cmd_ready && !map_valid) begin state<=IDLE;done<=1;end
        default: begin state<=IDLE;error<=1;end
      endcase
    end
  end
endmodule
