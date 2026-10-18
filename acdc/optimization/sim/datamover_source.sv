`timescale 1ns/1ps
// Behavioral MM2S source for addressed test memory, not a production IP model.
module datamover_source (
  input logic aclk,aresetn,
  input logic [71:0] cmd_data,
  input logic cmd_valid,
  output logic cmd_ready,
  output logic [63:0] data,
  output logic [7:0] keep,
  output logic last,valid,
  input logic ready,
  output logic [7:0] status_data,
  output logic status_valid,
  input logic status_ready,
  output integer supplied_words
);
  logic [63:0] memory[int unsigned];
  logic [31:0] address;
  logic [22:0] remaining;
  logic [3:0] tag;
  bit active=0,pending_status=0,held_valid=0;
  int fd,rc,nwords,status_delay=0;
  int unsigned key,rng=32'hadcd5a11;
  logic [63:0] value;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  initial begin
    supplied_words=0;valid=0;
    fd=$fopen("source_memory.txt","r");if(!fd)$fatal(1,"MM2S test memory missing");
    rc=$fscanf(fd,"%d\n",nwords);if(rc!=1)$fatal(1,"MM2S memory header");
    for(int j=0;j<nwords;j++)begin
      rc=$fscanf(fd,"%h %h\n",key,value);if(rc!=2)$fatal(1,"short MM2S memory");
      if(memory.exists(key))$fatal(1,"duplicate MM2S memory address");memory[key]=value;
    end
  end
  always @(negedge aclk)begin
    rng=next_rng(rng);cmd_ready=aresetn&&!active&&!pending_status&&(rng[0]||rng[4]);
    status_valid=aresetn&&pending_status&&status_delay==0;status_data=8'h80|tag;
    if(!aresetn)begin valid=0;held_valid=0;end
    else if(!held_valid)begin
      valid=active&&(rng[1]||rng[5]);
      if(valid)begin
        if(!memory.exists(address))$fatal(1,"read outside initialized DDR address=%h",address);
        data=memory[address];keep=8'hff;last=remaining==8;held_valid=1;
      end
    end
  end
  always @(posedge aclk)begin
    if(!aresetn)begin active=0;pending_status=0;held_valid=0;end
    else begin
      if(cmd_valid&&cmd_ready)begin
        if(active||pending_status||cmd_data[31:24]!=8'h40||!cmd_data[23]||cmd_data[22:0]==0||cmd_data[2:0]!=0)
          $fatal(1,"bad MM2S test command");
        address=cmd_data[63:32];remaining=cmd_data[22:0];tag=cmd_data[67:64];active=1;
      end
      if(valid&&ready)begin
        supplied_words++;held_valid=0;address+=8;remaining-=8;
        if(last)begin active=0;pending_status=1;status_delay=3;end
      end
      if(pending_status&&status_delay>0)status_delay--;
      if(status_valid&&status_ready)pending_status=0;
    end
  end
endmodule
