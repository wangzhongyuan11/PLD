// Eight independent byte lookups; two layer/branch tables, replicated per lane.
// Table index is the raw signed-int8 byte pattern, not a sign-extended address.
// Scheduler holds cfg_lock across a frame/layer interval to prevent live updates.
module map8 (
  input logic clk,resetn,cfg_lock,
  input logic cfg_table,
  input logic [7:0] cfg_index,cfg_value,
  input logic cfg_valid,
  output logic cfg_ready,
  input logic [63:0] s_data,
  input logic s_table,s_valid,
  output logic s_ready,
  output logic [63:0] m_data,
  output logic m_valid,
  input logic m_ready
);
  assign cfg_ready=resetn && !cfg_lock;
  assign s_ready=resetn && (!m_valid || m_ready);
  for(genvar lane=0;lane<8;lane++) begin: lanes
    (* ram_style="block" *) logic [7:0] lut [0:511];
    always_ff @(posedge clk) begin
      if(cfg_valid && cfg_ready) lut[{cfg_table,cfg_index}]<=cfg_value;
      if(s_valid && s_ready) m_data[lane*8+:8]<=lut[{s_table,s_data[lane*8+:8]}];
    end
  end
  always_ff @(posedge clk) begin
    if(!resetn) m_valid<=0;
    else if(s_ready) m_valid<=s_valid;
  end
endmodule
