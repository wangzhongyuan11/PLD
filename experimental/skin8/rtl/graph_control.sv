// Fetch 128-byte descriptors and optional 512-byte LUTs, then dispatch stripes.
// Reader is shared exclusively with this controller while engine_start is low
// and state is not WAIT_ENGINE. Descriptor ABI documented in DESCRIPTOR_ABI.md.
module graph_control (
  input logic clk,resetn,start,
  input logic [31:0] cfg_descriptors,
  input logic [15:0] cfg_count,
  output logic busy,done,error,
  output logic [15:0] completed,
  output logic [31:0] rd_addr,
  output logic [15:0] rd_words,
  output logic rd_cmd_valid,
  input logic rd_cmd_ready,
  input logic [63:0] rd_data,
  input logic rd_valid,rd_last,rd_error,
  output logic rd_ready,
  output logic engine_start,
  input logic engine_busy,engine_done,engine_error,
  output logic [1023:0] descriptor,
  output logic lut_valid,lut_table,
  output logic [7:0] lut_index,lut_value,
  input logic lut_ready
);
  typedef enum logic [3:0] {IDLE,CMD,DESC,CHECK,LCMD,LREAD,LAUNCH,WAIT_ENGINE,ADVANCE,ABORT} state_t;
  state_t state;
  logic [31:0] address;
  logic [15:0] remaining;
  logic [4:0] word_index;
  logic [6:0] lut_word_count;
  logic [9:0] lut_count;
  logic [3:0] buffered_bytes;
  logic [63:0] buffer_word;
  logic [32:0] requested_end;
  assign requested_end={1'b0,cfg_descriptors}+({17'b0,cfg_count}<<7);
  assign busy=state!=IDLE;
  assign rd_cmd_valid=state==CMD || state==LCMD;
  assign rd_ready=state==DESC || state==ABORT || (state==LREAD && buffered_bytes==0);
  assign engine_start=state==LAUNCH && !engine_busy;
  assign lut_valid=state==LREAD && buffered_bytes!=0;
  assign lut_table=lut_count[8];
  assign lut_index=lut_count[7:0];
  assign lut_value=buffer_word[7:0];
  always_ff @(posedge clk) begin
    if(!resetn) begin state<=IDLE;done<=0;error<=0;completed<=0;buffered_bytes<=0;end
    else begin
      done<=0;
      case(state)
        IDLE: if(start) begin
          error<=0;completed<=0;
          if(cfg_count==0 || cfg_descriptors[6:0]!=0 || requested_end>33'h100000000 || engine_busy) begin error<=1;done<=1;end
          else begin
            address<=cfg_descriptors;rd_addr<=cfg_descriptors;rd_words<=16;
            remaining<=cfg_count;word_index<=0;state<=CMD;
          end
        end
        CMD: if(rd_cmd_ready) state<=DESC;
        DESC: begin
          if(rd_valid && rd_ready) begin
            descriptor[word_index[3:0]*64+:64]<=rd_data;word_index<=word_index+1;
            if(rd_last!==(word_index==15)) begin error<=1;state<=ABORT;end
            else if(word_index==15) state<=CHECK;
          end
          if(rd_error) begin error<=1;state<=ABORT;end
        end
        CHECK: begin
          // Reserved bytes must be zero; upper geometry bits cannot be silently truncated.
          if(descriptor[288+:96]!=0 || descriptor[896+:128]!=0 ||
             descriptor[137+:7]!=0 || descriptor[153+:7]!=0 || descriptor[169+:7]!=0 || descriptor[185+:7]!=0 ||
             descriptor[197+:3]!=0 || descriptor[205+:3]!=0 || descriptor[210+:6]!=0 ||
             (descriptor[223] && (descriptor[256+:3]!=0 || {1'b0,descriptor[256+:32]}+33'd512>33'h100000000))) begin
            error<=1;state<=ABORT;
          end else if(descriptor[223]) begin
            rd_addr<=descriptor[256+:32];rd_words<=64;
            lut_count<=0;lut_word_count<=0;buffered_bytes<=0;state<=LCMD;
          end
          else state<=LAUNCH;
          if(rd_error) begin error<=1;state<=ABORT;end
        end
        LCMD: if(rd_cmd_ready) state<=LREAD;
        LREAD: begin
          if(rd_valid && rd_ready) begin
            buffer_word<=rd_data;buffered_bytes<=8;lut_word_count<=lut_word_count+1;
            if(rd_last!==(lut_word_count==63)) begin error<=1;state<=ABORT;end
          end
          if(lut_valid && lut_ready) begin
            buffer_word<=buffer_word>>8;buffered_bytes<=buffered_bytes-1;lut_count<=lut_count+1;
            if(lut_count==511) state<=LAUNCH;
          end
          if(rd_error) begin error<=1;state<=ABORT;end
        end
        LAUNCH: if(!engine_busy) state<=WAIT_ENGINE;
        WAIT_ENGINE: if(engine_done) begin
          if(engine_error) begin error<=1;done<=1;state<=IDLE;end
          else begin completed<=completed+1;state<=ADVANCE;end
        end
        ADVANCE: begin
          if(remaining==1) begin done<=1;state<=IDLE;end
          else begin
            remaining<=remaining-1;address<=address+128;rd_addr<=address+128;rd_words<=16;
            word_index<=0;state<=CMD;
          end
        end
        ABORT: if(rd_cmd_ready) begin buffered_bytes<=0;done<=1;state<=IDLE;end
        default: begin error<=1;state<=ABORT;end
      endcase
    end
  end
endmodule
