// Eight independent byte lookups, replicated small distributed tables.
// LUT writes occur outside active input execution. Invalid/padded lanes become 0.
module input_transform (
  input logic aclk,aresetn,
  input logic lut_we,
  input logic [8:0] lut_addr,
  input logic [7:0] lut_data,
  input logic [63:0] s_data,
  input logic [7:0] s_lane_mask,
  input logic [31:0] s_tag,
  input logic s_lookup,s_branch,s_zero,s_valid,
  output logic s_ready,
  output logic [63:0] m_data,
  output logic [31:0] m_tag,
  output logic m_valid,
  input logic m_ready,
  output logic invalid_int8
);
  logic advance;
  assign advance=!m_valid||m_ready;
  assign s_ready=aresetn&&advance&&!lut_we;
  for(genvar lane=0;lane<8;lane++)begin:lanes
    (* ram_style="distributed" *) logic [7:0] lut[512];
    always_ff @(posedge aclk)begin
      if(lut_we)lut[lut_addr]<=lut_data;
      if(s_valid&&s_ready)begin
        if(s_zero||!s_lane_mask[lane])m_data[lane*8+:8]<=0;
        else if(s_lookup)m_data[lane*8+:8]<=lut[{s_branch,s_data[lane*8+:8]}];
        else m_data[lane*8+:8]<=s_data[lane*8+:8];
      end
    end
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)begin m_valid<=0;invalid_int8<=0;end
    else begin
      if(advance)begin
        m_valid<=s_valid&&s_ready;
        if(s_valid&&s_ready)m_tag<=s_tag;
      end
      if(s_valid&&s_ready&&!s_zero)
        for(int lane=0;lane<8;lane++)if(s_lane_mask[lane]&&s_data[lane*8+:8]==8'h80)invalid_int8<=1;
    end
  end
endmodule
