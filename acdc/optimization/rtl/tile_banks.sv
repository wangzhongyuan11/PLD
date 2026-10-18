// Two independently owned tile slots, P spatial banks, eight INT8 Ci per word.
// Simple dual-port RAM: a full 64-bit read and byte-enabled 64-bit write.
// compute_active reserves its entire slot even during read gaps/backpressure.
module tile_banks #(
  parameter integer P=4,
  parameter integer WORDS=4608,
  parameter integer AW=$clog2(WORDS)
) (
  input logic aclk,aresetn,
  input logic compute_active,compute_slot,
  input logic load_slot,
  input logic [$clog2(P)-1:0] load_bank,
  input logic [AW-1:0] load_addr,
  input logic [63:0] load_data,
  input logic [7:0] load_keep,
  input logic load_valid,
  output logic load_ready,
  input logic [P*AW-1:0] s_addr,
  input logic [31:0] s_tag,
  input logic s_valid,
  output logic s_ready,
  output logic [P*64-1:0] m_data,
  output logic [31:0] m_tag,
  output logic m_valid,
  input logic m_ready,
  output logic address_error
);
  logic advance,read_enable;
  localparam integer PAGES=(WORDS+511)/512;
  logic [P*64-1:0] slot_data[2];
  logic read_slot;
  logic read_address_ok;
  assign advance=!m_valid||m_ready;
  assign load_ready=aresetn&&(!compute_active||load_slot!=compute_slot)&&load_addr<WORDS&&load_bank<P;
  always_comb begin
    read_address_ok=1;
    for(int b=0;b<P;b++)read_address_ok&=s_addr[b*AW+:AW]<WORDS;
  end
  assign s_ready=aresetn&&compute_active&&advance&&read_address_ok;
  assign read_enable=s_valid&&s_ready;
  assign m_data=slot_data[read_slot];
  for(genvar slot=0;slot<2;slot++)begin:slots
    for(genvar bank=0;bank<P;bank++)begin:banks
      logic [63:0] page_data[PAGES];
      logic [$clog2(PAGES+1)-1:0] selected_page;
      assign slot_data[slot][bank*64+:64]=page_data[selected_page];
      always_ff @(posedge aclk)begin
        if(read_enable&&compute_slot==slot)selected_page<=s_addr[bank*AW+:AW]/512;
      end
      // Explicit 512-word pages avoid Vivado's deep-RAM aspect-ratio packing,
      // which otherwise wastes 128 BRAM36 for this 72-BRAM payload geometry.
      for(genvar page=0;page<PAGES;page++)begin:pages
        (* ram_style="block" *) logic [63:0] mem[512];
        always_ff @(posedge aclk)begin
          if(load_valid&&load_ready&&load_slot==slot&&load_bank==bank&&load_addr/512==page)
            for(int byte_lane=0;byte_lane<8;byte_lane++)
              if(load_keep[byte_lane])mem[9'(load_addr)][byte_lane*8+:8]<=load_data[byte_lane*8+:8];
          if(read_enable&&compute_slot==slot&&s_addr[bank*AW+:AW]/512==page)
            page_data[page]<=mem[9'(s_addr[bank*AW+:AW])];
        end
      end
    end
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)begin m_valid<=0;address_error<=0;end
    else begin
      if(load_valid&&(load_addr>=WORDS||load_bank>=P))address_error<=1;
      if(s_valid&&compute_active&&!read_address_ok)address_error<=1;
      if(advance)begin
        m_valid<=read_enable;
        if(read_enable)begin read_slot<=compute_slot;m_tag<=s_tag;end
      end
    end
  end
endmodule
