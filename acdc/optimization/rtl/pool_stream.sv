// Ordered C8HW8 pixel stream -> exact signed 2x2/stride2 max pool.
// A tile must have even width/height, so rows never depend on another tile.
module pool_stream (
  input logic aclk,aresetn,start,
  input logic [15:0] cfg_width,cfg_height,
  input logic [63:0] s_data,
  input logic s_valid,
  output logic s_ready,
  output logic [63:0] m_data,
  output logic [7:0] m_x,m_y,
  output logic m_valid,
  input logic m_ready,
  output logic busy,done,error
);
  logic [15:0] width,height;
  logic [7:0] x,y;
  logic [63:0] left,horizontal,previous_row,horizontal_q;
  (* ram_style="block" *) logic [63:0] row_max[128];
  logic pair_valid,advance,input_done;
  logic [7:0] pair_x,pair_y;
  assign advance=!m_valid||m_ready;
  assign s_ready=aresetn&&busy&&!input_done&&advance;
  always_comb begin
    for(int c=0;c<8;c++)
      horizontal[c*8+:8]=($signed(left[c*8+:8])>$signed(s_data[c*8+:8]))?left[c*8+:8]:s_data[c*8+:8];
  end
  always_ff @(posedge aclk)begin
    if(s_valid&&s_ready&&x[0])begin
      if(!y[0])row_max[x[7:1]]<=horizontal;
      previous_row<=row_max[x[7:1]];
    end
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)begin busy<=0;done<=0;error<=0;m_valid<=0;pair_valid<=0;input_done<=0;end
    else begin
      done<=0;
      if(!busy&&start)begin
        error<=0;
        if(cfg_width==0||cfg_width>256||cfg_height==0||cfg_height>256||cfg_width[0]||cfg_height[0])begin
          error<=1;done<=1;
        end else begin
          width<=cfg_width;height<=cfg_height;x<=0;y<=0;
          busy<=1;input_done<=0;pair_valid<=0;m_valid<=0;
        end
      end
      if(advance)begin
        m_valid<=pair_valid;pair_valid<=0;
        if(pair_valid)begin
          for(int c=0;c<8;c++)m_data[c*8+:8]<=($signed(previous_row[c*8+:8])>$signed(horizontal_q[c*8+:8]))?previous_row[c*8+:8]:horizontal_q[c*8+:8];
          m_x<=pair_x;m_y<=pair_y;
        end
        if(s_valid&&s_ready)begin
          if(!x[0])left<=s_data;
          else if(y[0])begin
            horizontal_q<=horizontal;pair_x<={1'b0,x[7:1]};pair_y<={1'b0,y[7:1]};pair_valid<=1;
          end
          if(x+1==width)begin
            x<=0;
            if(y+1==height)input_done<=1;
            else y<=y+1;
          end else x<=x+1;
        end
      end
      if(m_valid&&m_ready&&m_x+1==width/2&&m_y+1==height/2)begin busy<=0;done<=1;end
    end
  end
endmodule
