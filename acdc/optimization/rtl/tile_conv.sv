// One Co8 spatial tile. Input is already halo-padded in tile_banks layout.
// Weights are Co-bank x (Ci8,Ky,Kx) 64-bit words. DMA/controller follows later.
module tile_conv #(
  parameter integer P=4,
  parameter integer WORDS=(P==4?4608:3584),
  parameter integer AW=$clog2(WORDS)
) (
  input logic aclk,aresetn,start,
  input logic [15:0] cfg_oh,cfg_ow,
  input logic [7:0] cfg_ci,
  input logic [3:0] cfg_co,
  input logic cfg_kernel3,cfg_relu,cfg_input_slot,cfg_weight_slot,
  output logic busy,done,
  output logic [2:0] error,
  input logic load_slot,
  input logic [$clog2(P)-1:0] load_bank,
  input logic [AW-1:0] load_addr,
  input logic [63:0] load_data,
  input logic [7:0] load_keep,
  input logic load_valid,
  output logic load_ready,
  input logic weight_slot,
  input logic [2:0] weight_bank,
  input logic [7:0] weight_addr,
  input logic [63:0] weight_data,
  input logic weight_valid,
  output logic weight_ready,
  input logic param_slot,param_valid,
  input logic [255:0] param_bias,param_multiplier,
  output logic param_ready,
  output logic [31:0] m_data,m_tag,
  output logic [3:0] m_keep,
  output logic m_first,m_last,m_valid,
  input logic m_ready
);
  typedef enum logic [2:0] {IDLE,SETUP,PLANE,CAPACITY,CHECK,RUN} state_t;
  state_t state;
  logic [15:0] oh,ow,ih,iw,columns,plane_words,x,y,row_base,ci_base,tap_row;
  logic [7:0] x_group;
  logic [8:0] column_numerator,column_count;
  logic [7:0] ci,tap;
  // Capacity validation is not a one-cycle cascaded multiply/compare path.
  // Separate registered products keep tile launch independent of DSP cascades.
  (* use_dsp="no" *) logic [31:0] capacity_words;
  logic [4:0] ci_group,ci_count;
  logic [3:0] co;
  logic [1:0] kx,ky;
  logic kernel3,relu,input_slot,wslot,issue_done;
  logic [255:0] biases[2],multipliers[2];
  logic [1:0] params_loaded;
  logic [511:0] weight_read[2];
  logic [511:0] compute_weights;
  logic [P*64-1:0] bank_data,compute_input;
  logic [P*AW-1:0] bank_addr;
  logic [P*AW-1:0] address_q;
  logic [31:0] address_tag;
  logic [7:0] address_tap,address_mask;
  logic [1:0] address_kx;
  logic address_valid,address_ready;
  logic [P*64-1:0] input_q;
  logic [511:0] weights_q;
  logic [P*8-1:0] lane_mask_q;
  logic [31:0] tag_q;
  logic input_valid;
  logic [31:0] bank_tag,request_tag;
  logic bank_valid,bank_ready,request_valid,request_ready,bank_error,compute_ready;
  logic [7:0] ci_mask;
  logic [1:0] response_kx;
  logic [P*8-1:0] lane_mask;
  logic compute_overflow;
  logic [2:0] queued_groups;
  integer pixel_offset,read_bank;
  assign busy=state!=IDLE;
  assign weight_ready=aresetn&&(!busy||weight_slot!=wslot)&&weight_addr<216;
  assign param_ready=aresetn&&(!busy||param_slot!=wslot);
  assign request_valid=state==RUN&&!issue_done;
  assign request_tag={14'd0,(ci_group==0&&kx==0&&ky==0),
                      (ci_group+1==ci_count&&kx==(kernel3?2:0)&&ky==(kernel3?2:0)),y[7:0],x[7:0]};
  // Elastic address and data boundaries preserve one request per cycle.
  // Coordinates no longer drive BRAM enables through address checks, and
  // the page/slot/rotation mux no longer feeds the DSP input directly.
  assign request_ready=!address_valid||address_ready;
  assign bank_ready=!input_valid||compute_ready;
  assign compute_weights=weight_read[wslot];
  // Validated iw <= 258. Keep constant division narrow even when P is 6;
  // a 32-bit expression here synthesizes a much longer reciprocal network.
  assign column_numerator=iw[8:0]+9'(P-1);
  assign column_count=column_numerator/9'(P);
  always_comb begin
    bank_addr=0;compute_input=0;lane_mask=0;
    // x always advances by P and starts at zero on each row. Track x/P
    // explicitly so the per-tap address path contains no divide or modulo.
    for(int b=0;b<P;b++)begin
      pixel_offset=b>=kx?b-kx:b+P-kx;
      if(x+pixel_offset<ow)
        bank_addr[b*AW+:AW]=AW'(ci_base+tap_row+{8'd0,x_group}+(b<kx?16'd1:16'd0));
    end
    for(int p=0;p<P;p++)begin
      read_bank=response_kx+p<P?response_kx+p:response_kx+p-P;
      for(int i=0;i<8;i++)if(ci_mask[i]&&bank_tag[7:0]+p<ow)
        compute_input[(p*8+i)*8+:8]=bank_data[read_bank*64+i*8+:8];
      for(int o=0;o<8;o++)lane_mask[p*8+o]=(bank_tag[7:0]+p<ow)&&(o<co);
    end
  end
  tile_banks #(.P(P),.WORDS(WORDS)) input_banks(
    .aclk,.aresetn,.compute_active(busy),.compute_slot(input_slot),
    .load_slot,.load_bank,.load_addr,.load_data,.load_keep,.load_valid,.load_ready,
    .s_addr(address_q),.s_tag(address_tag),.s_valid(address_valid),.s_ready(address_ready),
    .m_data(bank_data),.m_tag(bank_tag),.m_valid(bank_valid),.m_ready(bank_ready),.address_error(bank_error));
  for(genvar slot=0;slot<2;slot++)begin:weight_slots
    for(genvar bank=0;bank<8;bank++)begin:weight_banks
      (* ram_style="block" *) logic [63:0] mem[256];
      always_ff @(posedge aclk)begin
        if(weight_valid&&weight_ready&&weight_slot==slot&&weight_bank==bank)mem[weight_addr]<=weight_data;
        if(address_valid&&address_ready&&wslot==slot)weight_read[slot][bank*64+:64]<=mem[address_tap];
      end
    end
  end
  conv_compute #(.P(P)) compute(.aclk,.aresetn,
    .s_input(input_q),.s_weights(weights_q),.s_bias(biases[wslot]),.s_multiplier(multipliers[wslot]),
    .s_lane_mask(lane_mask_q),.s_tag({16'd0,tag_q[15:0]}),.s_relu(relu),
    .s_first(tag_q[17]),.s_last(tag_q[16]),.s_valid(input_valid),.s_ready(compute_ready),
    .m_data,.m_tag,.m_keep,.m_first,.m_last,.m_overflow(compute_overflow),.m_valid,.m_ready,.queued_groups);
  always_ff @(posedge aclk)begin
    if(!aresetn)begin state<=IDLE;done<=0;error<=0;params_loaded<=0;address_valid<=0;input_valid<=0;end
    else begin
      done<=0;
      if(param_valid&&param_ready)begin
        biases[param_slot]<=param_bias;multipliers[param_slot]<=param_multiplier;params_loaded[param_slot]<=1;
      end
      if(request_ready)begin
        address_valid<=request_valid;
        if(request_valid)begin
          address_q<=bank_addr;address_tag<=request_tag;address_tap<=tap;address_kx<=kx;
          for(int i=0;i<8;i++)address_mask[i]<=(ci_group*8+i)<ci;
        end
      end
      if(address_valid&&address_ready)begin
        response_kx<=address_kx;ci_mask<=address_mask;
      end
      if(bank_ready)begin
        input_valid<=bank_valid;
        if(bank_valid)begin
          input_q<=compute_input;weights_q<=compute_weights;
          lane_mask_q<=lane_mask;tag_q<=bank_tag;
        end
      end
      if(m_valid&&m_ready&&compute_overflow)error<=2;
      case(state)
        IDLE:if(start)begin
          error<=0;oh<=cfg_oh;ow<=cfg_ow;ci<=cfg_ci;co<=cfg_co;
          kernel3<=cfg_kernel3;relu<=cfg_relu;input_slot<=cfg_input_slot;wslot<=cfg_weight_slot;
          ih<=cfg_oh+(cfg_kernel3?2:0);iw<=cfg_ow+(cfg_kernel3?2:0);
          ci_count<=({1'b0,cfg_ci}+7)>>3;
          if(cfg_oh==0||cfg_oh>256||cfg_ow==0||cfg_ow>256||cfg_ci==0||cfg_ci>192||cfg_co==0||cfg_co>8||!params_loaded[cfg_weight_slot])begin
            error<=1;done<=1;
          end else state<=SETUP;
        end
        SETUP:begin columns<={7'd0,column_count};state<=PLANE;end
        PLANE:begin plane_words<=ih*columns;state<=CAPACITY;end
        CAPACITY:begin capacity_words<=plane_words*ci_count;state<=CHECK;end
        CHECK:begin
          if(capacity_words>WORDS)begin error<=1;done<=1;state<=IDLE;end
          else begin
            x<=0;x_group<=0;y<=0;kx<=0;ky<=0;ci_group<=0;tap<=0;
            row_base<=0;ci_base<=0;tap_row<=0;issue_done<=0;state<=RUN;
          end
        end
        RUN:begin
          if(request_valid&&request_ready)begin
            tap<=tap+1;
            if(kx<(kernel3?2:0))kx<=kx+1;
            else begin
              kx<=0;
              if(ky<(kernel3?2:0))begin ky<=ky+1;tap_row<=tap_row+columns;end
              else begin
                ky<=0;tap_row<=row_base;
                if(ci_group+1<ci_count)begin ci_group<=ci_group+1;ci_base<=ci_base+plane_words;end
                else begin
                  ci_group<=0;ci_base<=0;tap<=0;
                  if(x+P<ow)begin x<=x+P;x_group<=x_group+1;end
                  else if(y+1<oh)begin x<=0;x_group<=0;y<=y+1;row_base<=row_base+columns;tap_row<=row_base+columns;end
                  else issue_done<=1;
                end
              end
            end
          end
          if(m_valid&&m_ready&&m_last&&m_tag[15:8]==oh-1&&m_tag[7:0]+P>=ow)begin done<=1;state<=IDLE;end
          if(bank_error)error<=3;
        end
        default:state<=IDLE;
      endcase
    end
  end
endmodule
