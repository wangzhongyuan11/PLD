// Horizontal nearest2 expansion plus source-row cache for vertical repetition.
// cfg_replay emits a second copy without any DDR input. Cache is valid only
// after a successful complete load; reset invalidates it without clearing BRAM.
module nearest2_row (
  input logic clk,resetn,start,cfg_replay,
  input logic [8:0] cfg_width,
  output logic busy,done,error,
  input logic [63:0] s_data,
  input logic s_valid,
  output logic s_ready,
  output logic [63:0] m_data,
  output logic m_valid,m_last,
  input logic m_ready
);
  (* ram_style="block" *) logic [63:0] cache [0:255];
  logic replay,cache_valid,all_read,all_loaded;
  logic [8:0] width,cache_width,read_index,load_index;
  logic [63:0] read_data;
  logic read_valid,read_last,serial_ready;
  logic read_request;
  assign read_request=busy && replay && !all_read && (!read_valid || serial_ready);
  assign s_ready=resetn && busy && !replay && !all_loaded && serial_ready;
  pair_to_words duplicate(.clk,.resetn,.s_data(replay ? {read_data,read_data} : {s_data,s_data}),
    .s_second(1'b1),.s_last(replay ? read_last : load_index==width-1),
    .s_valid(busy && (replay ? read_valid : s_valid && !all_loaded)),.s_ready(serial_ready),.m_data,.m_last,.m_valid,.m_ready);
  always_ff @(posedge clk) begin
    if(s_valid && s_ready) cache[load_index[7:0]]<=s_data;
    if(read_request) read_data<=cache[read_index[7:0]];
  end
  always_ff @(posedge clk) begin
    if(!resetn) begin busy<=0;done<=0;error<=0;cache_valid<=0;read_valid<=0;end
    else begin
      done<=0;
      if(start && !busy) begin
        error<=0;
        if(cfg_width==0 || cfg_width>256 || (cfg_replay && (!cache_valid || cache_width!=cfg_width))) begin error<=1;done<=1;end
        else begin
          busy<=1;replay<=cfg_replay;width<=cfg_width;read_index<=0;load_index<=0;
          all_read<=0;all_loaded<=0;read_valid<=0;
          if(!cfg_replay) begin cache_valid<=0;cache_width<=cfg_width;end
        end
      end
      if(read_valid && serial_ready) read_valid<=0;
      if(read_request) begin
        read_valid<=1;read_last<=read_index==width-1;read_index<=read_index+1;
        if(read_index==width-1) all_read<=1;
      end
      if(s_valid && s_ready) begin
        load_index<=load_index+1;
        if(load_index==width-1) all_loaded<=1;
      end
      if(m_valid && m_ready && m_last) begin busy<=0;done<=1;if(!replay) cache_valid<=1;end
    end
  end
endmodule
