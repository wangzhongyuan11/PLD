// One-entry elastic fork. Each consumer acknowledges independently, exactly once.
module stream_fork #(
  parameter integer WIDTH=64
) (
  input logic aclk,aresetn,
  input logic [WIDTH-1:0] s_data,
  input logic s_second_enable,s_valid,
  output logic s_ready,
  output logic [WIDTH-1:0] m0_data,m1_data,
  output logic m0_valid,m1_valid,
  input logic m0_ready,m1_ready
);
  logic [WIDTH-1:0] held;
  logic pending0,pending1;
  assign m0_data=held;assign m1_data=held;
  assign m0_valid=pending0;assign m1_valid=pending1;
  assign s_ready=aresetn&&(!pending0||m0_ready)&&(!pending1||m1_ready);
  always_ff @(posedge aclk)begin
    if(!aresetn)begin pending0<=0;pending1<=0;end
    else if(s_valid&&s_ready)begin
      held<=s_data;pending0<=1;pending1<=s_second_enable;
    end else begin
      if(m0_ready)pending0<=0;
      if(m1_ready)pending1<=0;
    end
  end
endmodule
