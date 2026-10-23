`timescale 1ns/1ps
// Two behavioral DataMover directions over ONE evolving test DDR memory.
// Each read sees actual earlier RTL writes, never the expected/golden tensor.
// Every write checks its per-address version against an independent CPU oracle.
// Bounded address store for simulators without SV associative arrays.
// Same handshakes, RNG and versioned write oracle as datamover_memory.
// No oracle values are injected into the evolving DDR memory.
module datamover_memory_portable (
  input logic aclk,aresetn,
  input logic [71:0] read_cmd_data,write_cmd_data,
  input logic read_cmd_valid,write_cmd_valid,
  output logic read_cmd_ready,write_cmd_ready,
  output logic [63:0] read_data,
  output logic [7:0] read_keep,read_status_data,write_status_data,
  output logic read_last,read_valid,read_status_valid,write_status_valid,
  input logic read_ready,read_status_ready,write_status_ready,
  input logic [63:0] write_data,
  input logic [7:0] write_keep,
  input logic write_last,write_valid,
  output logic write_ready,
  output integer checked_words,expected_words,read_words,read_commands,write_commands
);
  localparam integer WORDS=1048576,SLOTS=65536;
  logic [63:0] memory[WORDS],expected[SLOTS],expected_key[SLOTS];
  int unsigned versions[WORDS];
  bit initialized[WORDS],version_present[WORDS],expected_present[SLOTS];
  int unsigned slot;
  function automatic int unsigned word_index(input logic [31:0] a);
    if(a<32'h10000000||a>=32'h10800000||a[2:0]!=0)$fatal(1,"portable DDR address outside 8MiB aperture: %h",a);
    return (a-32'h10000000)>>3;
  endfunction
  function automatic int unsigned lookup(input logic [63:0] key);
    int unsigned h,first;
    h=((key[31:3]*32'h9e3779b1)^(key[63:32]*32'h85ebca6b))&(SLOTS-1);first=h;
    while(expected_present[h]&&expected_key[h]!=key)begin
      h=(h+1)&(SLOTS-1);
      if(h==first)$fatal(1,"portable oracle table full");
    end
    return h;
  endfunction
  logic [31:0] ra,wa;logic [22:0] rr,wr;logic [3:0] rt,wt;
  bit reading=0,writing=0,rs=0,ws=0,held_read=0,stalled_write=0;
  logic [72:0] held_write;
  int rd=0,wd=0,fd,rc,nwords,version;
  int unsigned address,rng=32'hacdc2211;
  longint unsigned oracle_key;
  logic [63:0] value;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  initial begin
    checked_words=0;read_words=0;read_commands=0;write_commands=0;read_valid=0;
    fd=$fopen("source_memory.txt","r");if(!fd)$fatal(1,"DDR initial memory missing");
    rc=$fscanf(fd,"%d\n",nwords);
    for(int j=0;j<nwords;j++)begin
      rc=$fscanf(fd,"%h %h\n",address,value);
      if(rc!=2||initialized[word_index(address)])$fatal(1,"bad DDR initial memory");
      memory[word_index(address)]=value;initialized[word_index(address)]=1;
    end
    $fclose(fd);fd=$fopen("memory_expected.txt","r");if(!fd)$fatal(1,"DDR oracle missing");
    rc=$fscanf(fd,"%d\n",expected_words);
    if(expected_words>SLOTS/2)$fatal(1,"portable oracle capacity exceeded");
    for(int j=0;j<expected_words;j++)begin
      rc=$fscanf(fd,"%h %h %h\n",address,version,value);
      oracle_key={32'(version),address};
      slot=lookup(oracle_key);
      if(rc!=3||expected_present[slot])$fatal(1,"bad DDR version oracle");
      expected[slot]=value;expected_key[slot]=oracle_key;expected_present[slot]=1;
    end
    $fclose(fd);
  end
  always @(negedge aclk)begin
    rng=next_rng(rng);
    read_cmd_ready=aresetn&&!reading&&!rs&&(rng[0]||rng[4]);
    write_cmd_ready=aresetn&&!writing&&!ws&&(rng[1]||rng[5]);
    write_ready=aresetn&&writing&&(rng[2]||rng[6]);
    read_status_valid=aresetn&&rs&&rd==0;read_status_data=8'h80|rt;
    write_status_valid=aresetn&&ws&&wd==0;write_status_data=8'h80|wt;
    if(!aresetn)begin read_valid=0;held_read=0;end
    else if(!held_read)begin
      read_valid=reading&&(rng[3]||rng[7]);
      if(read_valid)begin
        if(!initialized[word_index(ra)])$fatal(1,"read before RTL produced DDR address=%h",ra);
        read_data=memory[word_index(ra)];read_keep=8'hff;read_last=rr==8;held_read=1;
      end
    end
  end
  always @(posedge aclk)begin
    if(!aresetn)begin reading=0;writing=0;rs=0;ws=0;held_read=0;stalled_write=0;end
    else begin
      if(stalled_write&&(!write_valid||{write_last,write_keep,write_data}!==held_write))$fatal(1,"DDR writer unstable under backpressure");
      stalled_write=write_valid&&!write_ready;held_write={write_last,write_keep,write_data};
      if(read_cmd_valid&&read_cmd_ready)begin
        if(read_cmd_data[31:23]!=9'h081||read_cmd_data[22:0]==0||read_cmd_data[2:0]!=0||read_cmd_data[34:32]!=0)$fatal(1,"bad MM2S command");
        ra=read_cmd_data[63:32];rr=read_cmd_data[22:0];rt=read_cmd_data[67:64];reading=1;read_commands++;
      end
      if(write_cmd_valid&&write_cmd_ready)begin
        if(write_cmd_data[31:23]!=9'h081||write_cmd_data[22:0]==0||write_cmd_data[2:0]!=0||write_cmd_data[34:32]!=0)$fatal(1,"bad S2MM command");
        wa=write_cmd_data[63:32];wr=write_cmd_data[22:0];wt=write_cmd_data[67:64];writing=1;write_commands++;
      end
      if(read_valid&&read_ready)begin
        read_words++;held_read=0;ra+=8;rr-=8;
        if(read_last)begin reading=0;rs=1;rd=3;end
      end
      if(write_valid&&write_ready)begin
        if(!writing||write_keep!=8'hff||write_last!=(wr==8))$fatal(1,"DDR write framing error");
        if(!version_present[word_index(wa)])begin versions[word_index(wa)]=0;version_present[word_index(wa)]=1;end
        oracle_key={versions[word_index(wa)],wa};slot=lookup(oracle_key);
        if(!expected_present[slot])$fatal(1,"unexpected DDR write addr=%h version=%0d",wa,versions[word_index(wa)]);
        if(write_data!==expected[slot])$fatal(1,"DDR mismatch addr=%h version=%0d actual=%h expected=%h",wa,versions[word_index(wa)],write_data,expected[slot]);
        memory[word_index(wa)]=write_data;initialized[word_index(wa)]=1;versions[word_index(wa)]++;checked_words++;wa+=8;wr-=8;
        if(write_last)begin writing=0;ws=1;wd=30+int'(rng[4:0]);end
      end
      if(rs&&rd>0)rd--;if(ws&&wd>0)wd--;
      if(read_status_valid&&read_status_ready)rs=0;
      if(write_status_valid&&write_status_ready)ws=0;
    end
  end
endmodule
