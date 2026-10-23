// Co8 tile output -> zero-filled C8HW8 pixels -> independent full/pool rows.
// Two row slots per stream hide DDR write latency behind ongoing convolution.
// cfg_*_address is the tile's first destination pixel, strides are full-image.
module tile_writer #(
  parameter integer P=4,
  parameter integer TIMEOUT_CYCLES=10000000
) (
  input logic aclk,aresetn,start,
  input logic [15:0] cfg_width,cfg_height,
  input logic [31:0] cfg_full_address,cfg_full_stride,cfg_pool_address,cfg_pool_stride,
  input logic cfg_pool,
  input logic [31:0] s_data,s_tag,
  input logic [3:0] s_keep,
  input logic s_first,s_last,s_valid,
  output logic s_ready,
  output logic [71:0] cmd_data,
  output logic cmd_valid,
  input logic cmd_ready,
  input logic [7:0] status_data,
  input logic status_valid,
  output logic status_ready,
  output logic [63:0] dma_data,
  output logic [7:0] dma_keep,
  output logic dma_last,dma_valid,
  input logic dma_ready,
  output logic busy,done,
  output logic [7:0] error
);
  typedef enum logic [1:0] {IDLE,ACTIVE,FAULT} state_t;
  typedef enum logic [1:0] {ROW_IDLE,ROW_REQUEST,ROW_TRANSFER} row_state_t;
  state_t state;row_state_t row_state;
  logic [15:0] width,height;
  logic pool_enabled,input_done;
  logic half,lower_valid;
  logic [3:0] pixel_index;
  logic [31:0] lower,lower_tag,clean_data;
  logic [79:0] pixel_data,full_data,fork_pool_data;
  logic pixel_valid,pixel_ready,full_valid,full_ready,pool_input_valid,pool_input_ready;
  logic [63:0] pool_data;
  logic [7:0] pool_x,pool_y;
  logic pool_valid,pool_ready,pool_busy,pool_done,pool_error;
  logic [1:0] full_rows,pool_rows;
  (* ram_style="block" *) logic [63:0] full_memory[512];
  (* ram_style="block" *) logic [63:0] pool_memory[256];
  logic [15:0] full_completed,pool_completed;
  logic [31:0] full_address,pool_address,full_stride,pool_stride;
  logic prefer_pool,active_pool,active_slot;
  logic [15:0] active_width;
  logic [31:0] active_address;
  logic [3:0] row_tag;
  logic [15:0] read_index;
  logic payload_valid,payload_last,payload_ready,payload_advance,read_enable;
  logic [63:0] full_read,pool_read,payload_data;
  logic transfer_ready,transfer_busy,transfer_done,transfer_fault;
  logic [7:0] transfer_error;
  logic [31:0] transfer_bytes;
  logic [8:0] pixel_x;
  assign busy=state!=IDLE;
  always_comb begin
    for(int c=0;c<4;c++)clean_data[c*8+:8]=s_keep[c]?s_data[c*8+:8]:8'd0;
  end
  assign pixel_x={1'b0,s_tag[7:0]}+pixel_index;
  assign s_ready=aresetn&&state==ACTIVE&&!input_done&&(!pixel_valid||pixel_ready);
  assign full_ready=state==ACTIVE&&!full_rows[full_data[72]];
  assign pool_ready=state==ACTIVE&&!pool_rows[pool_y[0]];
  stream_fork #(.WIDTH(80)) fork_output(.aclk,.aresetn,.s_data(pixel_data),.s_second_enable(pool_enabled),
    .s_valid(pixel_valid),.s_ready(pixel_ready),.m0_data(full_data),.m1_data(fork_pool_data),
    .m0_valid(full_valid),.m1_valid(pool_input_valid),.m0_ready(full_ready),.m1_ready(pool_input_ready));
  pool_stream pool(.aclk,.aresetn,.start(start&&state==IDLE&&cfg_pool),
    .cfg_width,.cfg_height,.s_data(fork_pool_data[63:0]),.s_valid(pool_input_valid),.s_ready(pool_input_ready),
    .m_data(pool_data),.m_x(pool_x),.m_y(pool_y),.m_valid(pool_valid),.m_ready(pool_ready),
    .busy(pool_busy),.done(pool_done),.error(pool_error));
  assign payload_advance=!payload_valid||payload_ready;
  assign read_enable=row_state!=ROW_IDLE&&payload_advance&&read_index<active_width;
  assign payload_data=active_pool?pool_read:full_read;
  always_ff @(posedge aclk)begin
    if(full_valid&&full_ready)full_memory[{full_data[72],full_data[71:64]}]<=full_data[63:0];
    if(pool_valid&&pool_ready)pool_memory[{pool_y[0],pool_x[6:0]}]<=pool_data;
    if(read_enable)begin
      if(active_pool)pool_read<=pool_memory[{active_slot,read_index[6:0]}];
      else full_read<=full_memory[{active_slot,read_index[7:0]}];
    end
  end
  dma_transfer #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) transfer(.aclk,.aresetn,
    .req_address(active_address),.req_bytes({4'd0,active_width,3'd0}),.req_tag(row_tag),
    .req_valid(row_state==ROW_REQUEST),.req_ready(transfer_ready),
    .cmd_data,.cmd_valid,.cmd_ready,.status_data,.status_valid,.status_ready,
    .s_data(payload_data),.s_keep(8'hff),.s_last(payload_last),.s_valid(payload_valid),.s_ready(payload_ready),
    .m_data(dma_data),.m_keep(dma_keep),.m_last(dma_last),.m_valid(dma_valid),.m_ready(dma_ready),
    .busy(transfer_busy),.done(transfer_done),.fault(transfer_fault),.error(transfer_error),.actual_bytes(transfer_bytes));
  always_ff @(posedge aclk)begin
    if(!aresetn)begin
      state<=IDLE;row_state<=ROW_IDLE;done<=0;error<=0;half<=0;pixel_index<=0;pixel_valid<=0;
      full_rows<=0;pool_rows<=0;full_completed<=0;pool_completed<=0;payload_valid<=0;input_done<=0;
    end else begin
      done<=0;
      if(pixel_valid&&pixel_ready)pixel_valid<=0;
      if(state==IDLE&&start)begin
        error<=0;
        if(cfg_width==0||cfg_width>256||cfg_height==0||cfg_height>256||cfg_full_address[2:0]!=0||
           cfg_full_stride[2:0]!=0||cfg_full_stride<cfg_width*8||
           (cfg_pool&&(cfg_width[0]||cfg_height[0]||cfg_pool_address[2:0]!=0||cfg_pool_stride[2:0]!=0||cfg_pool_stride<cfg_width*4)))begin
          error<=1;done<=1;state<=FAULT;
        end else begin
          state<=ACTIVE;width<=cfg_width;height<=cfg_height;pool_enabled<=cfg_pool;
          full_address<=cfg_full_address;pool_address<=cfg_pool_address;
          full_stride<=cfg_full_stride;pool_stride<=cfg_pool_stride;
          full_completed<=0;pool_completed<=0;full_rows<=0;pool_rows<=0;
          half<=0;pixel_index<=0;input_done<=0;pixel_valid<=0;
          row_state<=ROW_IDLE;prefer_pool<=0;row_tag<=0;payload_valid<=0;
        end
      end
      if(state==ACTIVE)begin
        if(s_valid&&s_ready)begin
          if(s_first&&(half||pixel_index!=0))error<=2;
          if(!half)begin lower<=clean_data;lower_valid<=|s_keep;lower_tag<=s_tag;half<=1;end
          else begin
            if(lower_tag!=s_tag)error<=2;
            half<=0;
            if(lower_valid)begin
              pixel_data<={s_tag[15:8],pixel_x[7:0],clean_data,lower};pixel_valid<=1;
            end
            if(s_last)begin
              if(pixel_index!=P-1)error<=2;
              pixel_index<=0;
              if(s_tag[15:8]==height-1&&s_tag[7:0]+P>=width)input_done<=1;
            end else pixel_index<=pixel_index+1;
          end
        end
        if(full_valid&&full_ready&&full_data[71:64]+1==width)full_rows[full_data[72]]<=1;
        if(pool_valid&&pool_ready&&pool_x+1==width/2)pool_rows[pool_y[0]]<=1;
        case(row_state)
          ROW_IDLE:begin
            if((prefer_pool&&pool_enabled&&pool_rows[pool_completed[0]])||
               (!full_rows[full_completed[0]]&&pool_enabled&&pool_rows[pool_completed[0]]))begin
              active_pool<=1;active_slot<=pool_completed[0];active_width<=width/2;active_address<=pool_address;
              row_state<=ROW_REQUEST;read_index<=0;payload_valid<=0;
            end else if(full_rows[full_completed[0]])begin
              active_pool<=0;active_slot<=full_completed[0];active_width<=width;active_address<=full_address;
              row_state<=ROW_REQUEST;read_index<=0;payload_valid<=0;
            end
          end
          ROW_REQUEST:if(transfer_ready)row_state<=ROW_TRANSFER;
          ROW_TRANSFER:if(transfer_done)begin
            if(transfer_fault)begin state<=FAULT;error<=8'h80|transfer_error;done<=1;end
            else begin
              if(active_pool)begin pool_rows[active_slot]<=0;pool_completed<=pool_completed+1;pool_address<=pool_address+pool_stride;end
              else begin full_rows[active_slot]<=0;full_completed<=full_completed+1;full_address<=full_address+full_stride;end
              prefer_pool<=!active_pool;row_tag<=row_tag+1;row_state<=ROW_IDLE;
            end
          end
          default:;
        endcase
        if(payload_advance)begin
          payload_valid<=read_enable;
          if(read_enable)begin payload_last<=read_index+1==active_width;read_index<=read_index+1;end
        end
        if(pool_error)error<=4;
        if(input_done&&full_completed==height&&(!pool_enabled||pool_completed==height/2)&&row_state==ROW_IDLE)begin
          state<=IDLE;done<=1;
        end
      end
    end
  end
endmodule
