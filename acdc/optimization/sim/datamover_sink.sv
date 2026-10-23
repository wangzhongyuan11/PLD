`timescale 1ns/1ps
// Behavioral S2MM test sink, not production IP. Every addressed 64-bit write
// must match the independent NumPy conv/pool oracle exactly, once only.
module datamover_sink (
  input logic aclk,aresetn,
  input logic [71:0] cmd_data,
  input logic cmd_valid,
  output logic cmd_ready,
  input logic [63:0] data,
  input logic [7:0] keep,
  input logic last,valid,
  output logic ready,
  output logic [7:0] status_data,
  output logic status_valid,
  input logic status_ready,
  output integer checked_words,expected_words
);
  logic [63:0] expected[int unsigned];
  bit seen[int unsigned];
  logic [31:0] address;
  logic [22:0] remaining;
  logic [3:0] tag;
  bit active=0,pending_status=0,stalled=0;
  logic [72:0] held;
  int status_delay=0,fd,rc;
  int unsigned key,rng=32'hacdc5a22;
  logic [63:0] value;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  initial begin
    checked_words=0;
    fd=$fopen("writer_expected.txt","r");if(!fd)$fatal(1,"writer oracle missing");
    rc=$fscanf(fd,"%d\n",expected_words);if(rc!=1)$fatal(1,"writer oracle header");
    for(int j=0;j<expected_words;j++)begin
      rc=$fscanf(fd,"%h %h\n",key,value);if(rc!=2)$fatal(1,"short writer oracle");
      if(expected.exists(key))$fatal(1,"overlapping writer oracle address %h",key);
      expected[key]=value;
    end
  end
  always @(negedge aclk)begin
    rng=next_rng(rng);cmd_ready=aresetn&&!active&&!pending_status&&(rng[0]||rng[5]);
    ready=aresetn&&active&&(rng[1]||rng[4]);
    status_valid=aresetn&&pending_status&&status_delay==0;
    status_data=8'h80|tag;
  end
  always @(posedge aclk)begin
    if(!aresetn)begin active=0;pending_status=0;stalled=0;end
    else begin
      if(stalled&&(!valid||{last,keep,data}!==held))$fatal(1,"writer DMA output unstable");
      stalled=valid&&!ready;held={last,keep,data};
      if(cmd_valid&&cmd_ready)begin
        if(active||pending_status||cmd_data[31:24]!=8'h40||!cmd_data[23]||cmd_data[22:0]==0||cmd_data[2:0]!=0)
          $fatal(1,"bad writer command");
        address=cmd_data[63:32];remaining=cmd_data[22:0];tag=cmd_data[67:64];active=1;
      end
      if(valid&&ready)begin
        if(!active||keep!=8'hff||last!=(remaining==8))$fatal(1,"writer packet framing");
        if(!expected.exists(address))$fatal(1,"unexpected DDR write address=%h",address);
        if(seen.exists(address))$fatal(1,"duplicate DDR write address=%h",address);
        if(data!==expected[address])$fatal(1,"DDR value mismatch address=%h actual=%h expected=%h",address,data,expected[address]);
        seen[address]=1;checked_words++;address+=8;remaining-=8;
        if(last)begin active=0;pending_status=1;status_delay=3;end
      end
      if(pending_status&&status_delay>0)status_delay--;
      if(status_valid&&status_ready)pending_status=0;
    end
  end
endmodule
