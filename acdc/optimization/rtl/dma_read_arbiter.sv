// Packet-atomic fair arbitration of independent DataMover MM2S clients.
// Owner is retained until BOTH its TLAST and status have been acknowledged.
module dma_read_arbiter #(
  parameter integer N=3,
  parameter integer OW=$clog2(N)
) (
  input logic aclk,aresetn,
  input logic [N*72-1:0] c_cmd_data,
  input logic [N-1:0] c_cmd_valid,
  output logic [N-1:0] c_cmd_ready,
  output logic [N*64-1:0] c_data,
  output logic [N*8-1:0] c_keep,
  output logic [N-1:0] c_last,c_valid,
  input logic [N-1:0] c_ready,
  output logic [N*8-1:0] c_status,
  output logic [N-1:0] c_status_valid,
  input logic [N-1:0] c_status_ready,
  output logic [71:0] cmd_data,
  output logic cmd_valid,
  input logic cmd_ready,
  input logic [63:0] data,
  input logic [7:0] keep,
  input logic last,valid,
  output logic ready,
  input logic [7:0] status,
  input logic status_valid,
  output logic status_ready,
  output logic busy,protocol_error
);
  logic [OW-1:0] owner,priority_owner,selected,locked_owner,target;
  logic found,data_done,status_done,locked;
  integer candidate;
  always_comb begin
    found=0;selected=0;
    for(int offset=0;offset<N;offset++)begin
      candidate=(priority_owner+offset)%N;
      if(!found&&c_cmd_valid[candidate])begin selected=OW'(candidate);found=1;end
    end
    target=locked?locked_owner:selected;
    cmd_data=c_cmd_data[target*72+:72];cmd_valid=aresetn&&!busy&&(locked?c_cmd_valid[target]:found);
    c_cmd_ready=0;c_valid=0;c_status_valid=0;c_last=0;ready=0;status_ready=0;
    for(int i=0;i<N;i++)begin c_data[i*64+:64]=data;c_keep[i*8+:8]=keep;c_status[i*8+:8]=status;end
    if(cmd_valid)c_cmd_ready[target]=cmd_ready;
    if(aresetn&&busy)begin
      if(!data_done)begin c_valid[owner]=valid;c_last[owner]=last;ready=c_ready[owner];end
      if(!status_done)begin c_status_valid[owner]=status_valid;status_ready=c_status_ready[owner];end
    end
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)begin busy<=0;priority_owner<=0;protocol_error<=0;data_done<=0;status_done<=0;locked<=0;end
    else begin
      if(cmd_valid&&!cmd_ready)begin locked<=1;locked_owner<=target;end
      if(cmd_valid&&cmd_ready)begin owner<=target;busy<=1;data_done<=0;status_done<=0;locked<=0;end
      if(busy)begin
        if(valid&&ready&&last)data_done<=1;
        if(status_valid&&status_ready)status_done<=1;
        if((data_done&&valid)||(status_done&&status_valid))protocol_error<=1;
        if(data_done&&status_done)begin busy<=0;priority_owner<=owner==N-1?0:owner+1;end
      end
    end
  end
  initial if(N<2)$fatal(1,"read arbiter requires at least two clients");
endmodule
