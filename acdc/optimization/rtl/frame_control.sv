// ACD8 / ABI 2.0 AXI4-Lite control. Independent AW/W capture, bounded response
// storage. Fault recovery requires common vendor/core quiesce/reset outside this
// register block; ACK_DONE and CLEAR_CACHE never pretend to reset active DMA.
module frame_control #(
  parameter integer P=4,
  parameter integer CLOCK_HZ=100000000
) (
  input logic aclk,aresetn,
  input logic [7:0] s_axi_awaddr,
  input logic s_axi_awvalid,
  output logic s_axi_awready,
  input logic [31:0] s_axi_wdata,
  input logic [3:0] s_axi_wstrb,
  input logic s_axi_wvalid,
  output logic s_axi_wready,
  output logic [1:0] s_axi_bresp,
  output logic s_axi_bvalid,
  input logic s_axi_bready,
  input logic [7:0] s_axi_araddr,
  input logic s_axi_arvalid,
  output logic s_axi_arready,
  output logic [31:0] s_axi_rdata,
  output logic [1:0] s_axi_rresp,
  output logic s_axi_rvalid,
  input logic s_axi_rready,
  output logic start,clear_cache,
  output logic [31:0] cfg_program,cfg_epoch,
  output logic [15:0] cfg_count,
  input logic busy,done,fault,
  input logic [15:0] error,submitted,completed,
  input logic [31:0] error_tag,cycles,input_hits,parameter_hits,overlap_cycles,
  output logic interrupt
);
  logic aw_pending,w_pending,done_latched,launch_pending;
  logic [7:0] awaddr;
  logic [31:0] wdata,masked_data;
  logic [3:0] wstrb;
  logic locked,write_ok;
  assign locked=busy||launch_pending;
  assign interrupt=done_latched||fault;
  assign s_axi_awready=aresetn&&!aw_pending&&!s_axi_bvalid;
  assign s_axi_wready=aresetn&&!w_pending&&!s_axi_bvalid;
  assign s_axi_arready=aresetn&&(!s_axi_rvalid||s_axi_rready);
  function automatic logic [31:0] merge_bytes(input logic [31:0] old_value,new_value,input logic [3:0] strobe);
    for(int i=0;i<4;i++)merge_bytes[i*8+:8]=strobe[i]?new_value[i*8+:8]:old_value[i*8+:8];
  endfunction
  function automatic logic readable(input logic [7:0] a);
    return a[1:0]==0&&a<=8'h3c;
  endfunction
  function automatic logic [31:0] read_register(input logic [7:0] a);
    case(a)
      8'h00:return 32'h41434438;
      8'h04:return 32'h00020000;
      8'h08:return {16'h001f,8'd8,8'(P)};
      8'h0c:return 0;
      8'h10:return {28'd0,launch_pending,fault,done_latched,locked};
      8'h14:return cfg_program;
      8'h18:return {16'd0,cfg_count};
      8'h1c:return cfg_epoch;
      8'h20:return {16'd0,error};
      8'h24:return error_tag;
      8'h28:return cycles;
      8'h2c:return {completed,submitted};
      8'h30:return input_hits;
      8'h34:return parameter_hits;
      8'h38:return overlap_cycles;
      8'h3c:return CLOCK_HZ;
      default:return 0;
    endcase
  endfunction
  always_comb begin
    masked_data=merge_bytes(0,wdata,wstrb);write_ok=awaddr[1:0]==0;
    case(awaddr)
      8'h0c:begin
        write_ok&=masked_data==0||masked_data==1||masked_data==2||masked_data==4;
        if(masked_data[1:0]!=0)write_ok&=!locked&&!fault;
      end
      8'h14,8'h1c:write_ok&=!locked&&!fault;
      8'h18:write_ok&=!locked&&!fault&&masked_data[31:16]==0;
      default:write_ok=0;
    endcase
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)begin
      aw_pending<=0;w_pending<=0;s_axi_bvalid<=0;s_axi_bresp<=0;s_axi_rvalid<=0;s_axi_rresp<=0;s_axi_rdata<=0;
      cfg_program<=0;cfg_epoch<=0;cfg_count<=0;start<=0;clear_cache<=0;done_latched<=0;launch_pending<=0;
    end else begin
      start<=0;clear_cache<=0;
      if(busy)launch_pending<=0;
      if(done)begin done_latched<=1;launch_pending<=0;end
      if(s_axi_awvalid&&s_axi_awready)begin awaddr<=s_axi_awaddr;aw_pending<=1;end
      if(s_axi_wvalid&&s_axi_wready)begin wdata<=s_axi_wdata;wstrb<=s_axi_wstrb;w_pending<=1;end
      if(s_axi_bvalid&&s_axi_bready)s_axi_bvalid<=0;
      if(aw_pending&&w_pending&&!s_axi_bvalid)begin
        aw_pending<=0;w_pending<=0;s_axi_bvalid<=1;s_axi_bresp<=write_ok?2'b00:2'b10;
        if(write_ok)case(awaddr)
          8'h0c:begin
            if(masked_data[0])begin start<=1;launch_pending<=1;done_latched<=0;end
            if(masked_data[1])clear_cache<=1;
            if(masked_data[2]&&!done)done_latched<=0;
          end
          8'h14:cfg_program<=merge_bytes(cfg_program,wdata,wstrb);
          8'h18:cfg_count<=16'(merge_bytes({16'd0,cfg_count},wdata,wstrb));
          8'h1c:cfg_epoch<=merge_bytes(cfg_epoch,wdata,wstrb);
          default:;
        endcase
      end
      if(s_axi_rvalid&&s_axi_rready)s_axi_rvalid<=0;
      if(s_axi_arvalid&&s_axi_arready)begin
        s_axi_rvalid<=1;s_axi_rdata<=read_register(s_axi_araddr);s_axi_rresp<=readable(s_axi_araddr)?2'b00:2'b10;
      end
    end
  end
endmodule
