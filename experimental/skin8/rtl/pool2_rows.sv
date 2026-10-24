// Streaming signed 2x2 max pool for two consecutive NCHWc8 source rows.
// First row is reduced horizontally into BRAM; the second row produces outputs.
module pool2_rows (
  input logic clk,resetn,start,
  input logic [9:0] cfg_width,
  output logic busy,done,error,
  input logic [63:0] s_data,
  input logic s_valid,
  output logic s_ready,
  output logic [63:0] m_data,
  output logic m_valid,m_last,
  input logic m_ready
);
  (* ram_style="block" *) logic [63:0] row_mem [0:255];
  logic [63:0] left_pixel,top_pair,horizontal;
  logic [9:0] width,x;
  logic row,input_done;
  assign s_ready=resetn && busy && !input_done && (!row || !x[0] || !m_valid || m_ready);
  always_comb begin
    for(int lane=0;lane<8;lane++)
      horizontal[lane*8+:8]=$signed(left_pixel[lane*8+:8])>$signed(s_data[lane*8+:8]) ? left_pixel[lane*8+:8] : s_data[lane*8+:8];
  end
  always_ff @(posedge clk) begin
    if(s_valid && s_ready) begin
      if(!row && x[0]) row_mem[x[8:1]]<=horizontal;
      if(row && !x[0]) top_pair<=row_mem[x[8:1]];
    end
  end
  always_ff @(posedge clk) begin
    if(!resetn) begin busy<=0;done<=0;error<=0;m_valid<=0;input_done<=0;end
    else begin
      done<=0;
      if(m_valid && m_ready) begin
        m_valid<=0;
        if(m_last) begin busy<=0;done<=1;end
      end
      if(start && !busy) begin
        error<=0;
        if(cfg_width==0 || cfg_width>512 || cfg_width[0]) begin error<=1;done<=1;end
        else begin width<=cfg_width;x<=0;row<=0;busy<=1;input_done<=0;end
      end
      if(s_valid && s_ready) begin
        if(!x[0]) left_pixel<=s_data;
        else if(row) begin
          for(int lane=0;lane<8;lane++)
            m_data[lane*8+:8]<=$signed(top_pair[lane*8+:8])>$signed(horizontal[lane*8+:8]) ? top_pair[lane*8+:8] : horizontal[lane*8+:8];
          m_valid<=1;m_last<=x==width-1;
        end
        if(x==width-1) begin x<=0;row<=1;if(row) input_done<=1;end
        else x<=x+1;
      end
    end
  end
endmodule
