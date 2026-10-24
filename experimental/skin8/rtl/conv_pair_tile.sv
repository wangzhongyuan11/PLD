// One output-channel block (8 channels) over an explicitly halo-padded tile.
// Inputs and weights remain resident through all spatial outputs. The next
// feature page can load concurrently. No CPU involvement between dot products.
// cfg_iw/ih include halo; kernel is 1 or 3. ci_groups is ceil(Ci/8).
// Feature/weight padding lanes must be initialized to zero by the producer.
module conv_pair_tile #(parameter integer AW=12) (
  input logic clk,resetn,
  input logic start, cfg_page,cfg_k3,
  input logic [8:0] cfg_iw,cfg_ih,
  input logic [4:0] cfg_ci_groups,
  input logic [255:0] cfg_bias,
  output logic busy,error,done,
  input logic f_page,f_bank,
  input logic [AW-1:0] f_addr,
  input logic [63:0] f_data,
  input logic f_valid,
  output logic f_ready,
  input logic [2:0] w_lane,
  input logic [7:0] w_addr,
  input logic [63:0] w_data,
  input logic w_valid,
  output logic w_ready,
  output logic [511:0] m_acc,
  output logic [15:0] m_tag,
  output logic m_last,m_second_valid,m_overflow,m_valid,
  input logic m_ready
);
  logic page,k3,issuing;
  logic [4:0] groups,cg;
  logic [8:0] ow,oh,x,y;
  logic [1:0] kx,ky;
  logic [AW-1:0] stride,row_base,pixel_base,krow;
  logic [7:0] weight_index;
  logic [15:0] tag_counter,total_pairs;
  logic [255:0] bias;
  logic [1:0] locks;
  logic [AW-1:0] even_addr,odd_addr;
  logic read_ready,feat_valid,mac_ready;
  logic [127:0] feat_data,mac_x;
  logic first_q,last_q,second_q;
  logic [15:0] tag_q;
  logic [511:0] weights_q;
  logic [31:0] config_words;
  logic [31:0] config_area;
  logic [8:0] config_halfwidth,config_height;
  logic [2:0] prepare;
  logic [8:0] config_ow,config_oh;
  logic accepted;
  assign config_ow=cfg_iw-(cfg_k3 ? 2 : 0);
  assign config_oh=cfg_ih-(cfg_k3 ? 2 : 0);
  assign locks=busy ? (page ? 2'b10 : 2'b01) : 0;
  assign even_addr=pixel_base+krow+cg+(kx!=0 ? groups : 0);
  assign odd_addr=pixel_base+krow+cg+(kx==2 ? groups : 0);
  assign accepted=issuing && read_ready;
  assign w_ready=resetn && !busy;
  assign mac_x={second_q ? feat_data[127:64] : 64'b0,feat_data[63:0]};
  assign m_last=m_tag==total_pairs-1;
  // All rows have equal pair count; odd final x has only one real output.
  // Carry this bit through the MAC tag rather than calculating division here.
  logic [16:0] mac_tag;
  assign m_tag=mac_tag[15:0];
  assign m_second_valid=mac_tag[16];
  feature_banks #(.AW(AW)) storage (
    .clk,.resetn,.page_locked(locks),
    .w_page(f_page),.w_bank(f_bank),.w_addr(f_addr),.w_data(f_data),.w_valid(f_valid),.w_ready(f_ready),
    .r_page(page),.r_swap(kx==1),.r_even_addr(even_addr),.r_odd_addr(odd_addr),
    .r_valid(issuing),.r_ready(read_ready),.m_data(feat_data),.m_valid(feat_valid),.m_ready(mac_ready)
  );
  for(genvar o=0;o<8;o++) begin: weight_banks
    (* ram_style="block" *) logic [63:0] mem [0:255];
    always_ff @(posedge clk) begin
      if(w_valid && w_ready && w_lane==o) mem[w_addr]<=w_data;
      if(accepted) weights_q[o*64+:64]<=mem[weight_index];
    end
  end
  dot128 #(.TAG_W(17)) mac (
    .clk,.resetn,.s_x(mac_x),.s_w(weights_q),.s_bias(bias),.s_tag({second_q,tag_q}),
    .s_first(first_q),.s_last(last_q),.s_valid(feat_valid),.s_ready(mac_ready),
    .m_acc,.m_tag(mac_tag),.m_overflow,.m_valid,.m_ready
  );
  always_ff @(posedge clk) begin
    if(!resetn) begin busy<=0;issuing<=0;error<=0;done<=0;prepare<=0;end
    else begin
      done<=0;
      if(start && !busy) begin
        error<=0;
        if(cfg_ci_groups==0 || cfg_ci_groups>24 || cfg_iw==0 || cfg_ih==0 ||
           (cfg_k3 && (cfg_iw<3 || cfg_ih<3))) begin
          error<=1;
        end else begin
          busy<=1;issuing<=0;prepare<=1;page<=cfg_page;k3<=cfg_k3;
          groups<=cfg_ci_groups;bias<=cfg_bias;ow<=config_ow;oh<=config_oh;
          config_halfwidth<=({1'b0,cfg_iw}+10'd1)>>1;config_height<=cfg_ih;
          cg<=0;kx<=0;ky<=0;x<=0;y<=0;weight_index<=0;
          row_base<=0;pixel_base<=0;krow<=0;tag_counter<=0;
        end
      end
      if(prepare==1) begin
        config_area<=config_halfwidth*config_height;
        stride<=config_halfwidth*groups;
        total_pairs<=(({1'b0,ow}+10'd1)>>1)*oh;
        prepare<=2;
      end else if(prepare==2) begin
        config_words<=config_area*groups;prepare<=3;
      end else if(prepare==3) begin
        prepare<=0;
        if(config_words>(1<<AW)) begin busy<=0;error<=1;end
        else issuing<=1;
      end
      if(accepted) begin
        first_q<=weight_index==0;
        last_q<=cg==groups-1 && kx==(k3?2:0) && ky==(k3?2:0);
        second_q<=x+1<ow;tag_q<=tag_counter;
        weight_index<=weight_index+1;
        if(cg!=groups-1) cg<=cg+1;
        else begin
          cg<=0;
          if(kx!=(k3?2:0)) kx<=kx+1;
          else begin
            kx<=0;
            if(ky!=(k3?2:0)) begin ky<=ky+1;krow<=krow+stride;end
            else begin
              ky<=0;krow<=0;weight_index<=0;tag_counter<=tag_counter+1;
              if(x+2<ow) begin x<=x+2;pixel_base<=pixel_base+groups;end
              else begin
                x<=0;y<=y+1;row_base<=row_base+stride;pixel_base<=row_base+stride;
                if(y==oh-1) issuing<=0;
              end
            end
          end
        end
      end
      if(m_valid && m_ready && m_last) begin busy<=0;done<=1;end
    end
  end
endmodule
