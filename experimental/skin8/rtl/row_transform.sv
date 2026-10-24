// Layer frontend row primitive: normal, maxpool2, nearest2-load, nearest2-replay.
// Width is the source-row width. LUT follows the spatial operator exactly.
module row_transform (
  input logic clk,resetn,start,
  input logic [1:0] cfg_mode,
  input logic [9:0] cfg_width,
  input logic cfg_table,
  output logic busy,done,error,
  input logic lut_table,
  input logic [7:0] lut_index,lut_value,
  input logic lut_valid,
  output logic lut_ready,
  input logic [63:0] s_data,
  input logic s_valid,
  output logic s_ready,
  output logic [63:0] m_data,
  output logic m_valid,m_last,
  input logic m_ready
);
  logic [1:0] mode;
  logic [9:0] width,count;
  logic table_sel,normal_done;
  logic [63:0] pool_data,near_data,op_data;
  logic pool_valid,pool_last,pool_ready,pool_in_ready,pool_busy,pool_done,pool_error;
  logic near_valid,near_last,near_ready,near_in_ready,near_busy,near_done,near_error;
  logic op_valid,op_last,map_ready;
  pool2_rows pool(.clk,.resetn,.start(start && !busy && cfg_mode==1),.cfg_width,
    .busy(pool_busy),.done(pool_done),.error(pool_error),.s_data,.s_valid(s_valid && busy && mode==1),.s_ready(pool_in_ready),
    .m_data(pool_data),.m_valid(pool_valid),.m_last(pool_last),.m_ready(pool_ready));
  nearest2_row nearest(.clk,.resetn,.start(start && !busy && cfg_mode>=2 && cfg_width<=256),.cfg_replay(cfg_mode==3),.cfg_width(cfg_width[8:0]),
    .busy(near_busy),.done(near_done),.error(near_error),.s_data,.s_valid(s_valid && busy && mode==2),.s_ready(near_in_ready),
    .m_data(near_data),.m_valid(near_valid),.m_last(near_last),.m_ready(near_ready));
  assign op_data=mode==0 ? s_data : mode==1 ? pool_data : near_data;
  assign op_valid=busy && (mode==0 ? s_valid && !normal_done : mode==1 ? pool_valid : near_valid);
  assign op_last=mode==0 ? count==width-1 : mode==1 ? pool_last : near_last;
  assign pool_ready=busy && mode==1 && map_ready;
  assign near_ready=busy && mode>=2 && map_ready;
  assign s_ready=busy && (mode==0 ? map_ready && !normal_done : mode==1 ? pool_in_ready : mode==2 ? near_in_ready : 1'b0);
  map8 mapper(.clk,.resetn,.cfg_lock(busy),.cfg_table(lut_table),.cfg_index(lut_index),.cfg_value(lut_value),.cfg_valid(lut_valid),.cfg_ready(lut_ready),
    .s_data(op_data),.s_table(table_sel),.s_valid(op_valid),.s_ready(map_ready),.m_data,.m_valid,.m_ready);
  always_ff @(posedge clk) begin
    if(!resetn) begin busy<=0;done<=0;error<=0;end
    else begin
      done<=0;
      if(start && !busy) begin
        error<=0;
        if(cfg_width==0 || cfg_width>512 || (cfg_mode>=2 && cfg_width>256)) begin error<=1;done<=1;end
        else begin busy<=1;mode<=cfg_mode;width<=cfg_width;count<=0;normal_done<=0;table_sel<=cfg_table;end
      end
      if(op_valid && map_ready) begin
        m_last<=op_last;
        if(mode==0) begin count<=count+1;if(op_last) normal_done<=1;end
      end
      if(busy && ((mode==1 && pool_error) || (mode>=2 && near_error))) begin busy<=0;error<=1;done<=1;end
      if(m_valid && m_ready && m_last) begin busy<=0;done<=1;end
    end
  end
endmodule
