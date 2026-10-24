// AXI-Lite control. AW and W may arrive independently; one response outstanding.
// No software reset while AXI work is active. Global reset belongs to the BD.
module frame_registers (
  input logic clk,resetn,
  input logic [7:0] s_awaddr,
  input logic s_awvalid,
  output logic s_awready,
  input logic [31:0] s_wdata,
  input logic [3:0] s_wstrb,
  input logic s_wvalid,
  output logic s_wready,
  output logic [1:0] s_bresp,
  output logic s_bvalid,
  input logic s_bready,
  input logic [7:0] s_araddr,
  input logic s_arvalid,
  output logic s_arready,
  output logic [31:0] s_rdata,
  output logic [1:0] s_rresp,
  output logic s_rvalid,
  input logic s_rready,
  output logic start,
  output logic [31:0] cfg_descriptors,
  output logic [15:0] cfg_count,
  input logic busy,done,error,
  input logic [15:0] completed
);
  logic aw_pending,w_pending;
  logic [7:0] address;
  logic [31:0] data;
  logic [3:0] strobe;
  logic done_sticky,error_sticky;
  logic [31:0] cycles;
  assign s_awready=resetn && !aw_pending && !s_bvalid;
  assign s_wready=resetn && !w_pending && !s_bvalid;
  assign s_arready=resetn && !s_rvalid;
  function automatic logic [31:0] merge(input logic [31:0] old_value,new_value,input logic [3:0] enables);
    for(int b=0;b<4;b++) merge[b*8+:8]=enables[b] ? new_value[b*8+:8] : old_value[b*8+:8];
  endfunction
  always_ff @(posedge clk) begin
    if(!resetn) begin
      aw_pending<=0;w_pending<=0;s_bvalid<=0;s_rvalid<=0;start<=0;
      cfg_descriptors<=0;cfg_count<=0;done_sticky<=0;error_sticky<=0;cycles<=0;
    end else begin
      start<=0;
      if(busy) cycles<=cycles+1;
      if(done) begin done_sticky<=1;error_sticky<=error;end
      if(s_awvalid && s_awready) begin address<=s_awaddr;aw_pending<=1;end
      if(s_wvalid && s_wready) begin data<=s_wdata;strobe<=s_wstrb;w_pending<=1;end
      if(s_bvalid && s_bready) s_bvalid<=0;
      if(aw_pending && w_pending && !s_bvalid) begin
        aw_pending<=0;w_pending<=0;s_bvalid<=1;s_bresp<=0;
        if(address[1:0]!=0 || busy || start) s_bresp<=2;
        else case(address)
          8'h00: if(strobe[0] && data[0]) begin start<=1;done_sticky<=0;error_sticky<=0;cycles<=0;end
          8'h10: cfg_descriptors<=merge(cfg_descriptors,data,strobe);
          8'h14: begin
            if(strobe[0]) cfg_count[7:0]<=data[7:0];
            if(strobe[1]) cfg_count[15:8]<=data[15:8];
          end
          default: s_bresp<=2;
        endcase
      end
      if(s_rvalid && s_rready) s_rvalid<=0;
      if(s_arvalid && s_arready) begin
        s_rvalid<=1;s_rresp<=0;s_rdata<=0;
        if(s_araddr[1:0]!=0) s_rresp<=2;
        else case(s_araddr)
          8'h00: s_rdata<={28'b0,error_sticky,done_sticky,(busy || start),1'b0};
          8'h04: s_rdata<=32'h534b3800;
          8'h10: s_rdata<=cfg_descriptors;
          8'h14: s_rdata<={16'b0,cfg_count};
          8'h18: s_rdata<=cycles;
          8'h1c: s_rdata<={16'b0,completed};
          default: s_rresp<=2;
        endcase
      end
    end
  end
endmodule
