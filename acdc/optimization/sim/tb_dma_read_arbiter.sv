`timescale 1ns/1ps
module tb_dma_read_arbiter;
  localparam integer N=3;
  logic aclk=0,aresetn=0;
  logic [N*72-1:0] c_cmd_data;
  logic [N-1:0] c_cmd_valid=0,c_cmd_ready,c_last,c_valid,c_ready=0,c_status_valid,c_status_ready=0;
  logic [N*64-1:0] c_data;
  logic [N*8-1:0] c_keep,c_status;
  logic [71:0] cmd_data;
  logic cmd_valid,cmd_ready,last,valid,ready,status_valid,status_ready,busy,protocol_error;
  logic [63:0] data;
  logic [7:0] keep,status;
  logic [63:0] memory[int unsigned];
  int fd,rc,nwords,supplied_words,total=0,cycles=0;
  int completed[N]='{default:0},offset[N]='{default:0};
  bit active[N]='{default:0},data_done[N]='{default:0},status_done[N]='{default:0};
  int unsigned base[N]='{32'h08000000,32'h08010000,32'h08020000};
  int unsigned key,rng=32'h5acd0223;
  logic [63:0] value;
  logic [71:0] held_cmd;
  bit stalled=0;
  dma_read_arbiter #(.N(N)) dut(.*);
  datamover_source source(.aclk,.aresetn,.cmd_data,.cmd_valid,.cmd_ready,.data,.keep,.last,.valid,.ready,
    .status_data(status),.status_valid,.status_ready,.supplied_words);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk)begin
    rng=next_rng(rng);
    for(int i=0;i<N;i++)begin
      c_cmd_valid[i]=aresetn&&!active[i]&&completed[i]<10;
      c_cmd_data[i*72+:72]={4'd0,4'(i+completed[i]*3),base[i],8'h40,1'b1,23'((8+completed[i]%3)*8)};
      c_ready[i]=aresetn&&(cycles%300>80)&&(rng[i]||rng[i+4]);
      c_status_ready[i]=aresetn&&(rng[i+1]||rng[i+5]);
    end
  end
  always @(posedge aclk)if(aresetn)begin
    cycles++;
    if(stalled&&(!cmd_valid||cmd_data!==held_cmd))$fatal(1,"arbiter changed blocked command");
    stalled=cmd_valid&&!cmd_ready;held_cmd=cmd_data;
    if(!$onehot0(c_cmd_ready)||!$onehot0(c_valid)||!$onehot0(c_status_valid))$fatal(1,"arbiter not one-hot");
    if(protocol_error)$fatal(1,"arbiter protocol fault");
    for(int i=0;i<N;i++)begin
      if(c_cmd_valid[i]&&c_cmd_ready[i])begin
        if(active[i])$fatal(1,"client double command");active[i]=1;offset[i]=0;data_done[i]=0;status_done[i]=0;
      end
      if(c_valid[i]&&c_ready[i])begin
        key=base[i]+offset[i]*8;
        if(!active[i]||!memory.exists(key)||c_data[i*64+:64]!==memory[key]||c_keep[i*8+:8]!=8'hff)
          $fatal(1,"arbiter payload misrouted client=%0d offset=%0d",i,offset[i]);
        if(c_last[i]!=(offset[i]+1==8+completed[i]%3))$fatal(1,"arbiter TLAST");
        offset[i]++;if(c_last[i])data_done[i]=1;
      end
      if(c_status_valid[i]&&c_status_ready[i])begin
        if(!active[i]||c_status[i*8+:8]!=(8'h80|4'(i+completed[i]*3)))$fatal(1,"arbiter status misrouted");
        status_done[i]=1;
      end
      if(active[i]&&data_done[i]&&status_done[i])begin active[i]=0;completed[i]++;total++;end
    end
  end
  initial begin
    fd=$fopen("source_memory.txt","r");if(!fd)$fatal(1,"arbiter test memory missing");
    rc=$fscanf(fd,"%d\n",nwords);
    for(int j=0;j<nwords;j++)begin rc=$fscanf(fd,"%h %h\n",key,value);if(rc!=2)$fatal(1,"bad memory");memory[key]=value;end
    repeat(4)@(negedge aclk);aresetn=1;
    wait(total==30);repeat(5)@(negedge aclk);
    if(busy||protocol_error)$fatal(1,"arbiter did not drain");
    $display("TB_PASS dma_read_arbiter clients=3 packets=%0d supplied_words=%0d command_stability_fair_routing=PASS",total,supplied_words);$finish;
  end
  initial begin #2000000;$fatal(1,"read arbiter watchdog");end
endmodule
