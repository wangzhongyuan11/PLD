`timescale 1ns/1ps
module tb_tile_banks #(parameter integer P=4);
  localparam integer WORDS=1050,AW=$clog2(WORDS);
  logic aclk=0,aresetn=0,compute_active=0,compute_slot=0,load_slot=0;
  logic [$clog2(P)-1:0] load_bank=0;
  logic [AW-1:0] load_addr=0;
  logic [63:0] load_data=0;
  logic [7:0] load_keep=0;
  logic load_valid=0,load_ready;
  logic [P*AW-1:0] s_addr=0;
  logic [31:0] s_tag=0,m_tag;
  logic s_valid=0,s_ready,m_valid,m_ready=0,address_error;
  logic [P*64-1:0] m_data;
  logic [63:0] model[2][P][WORDS];
  logic [P*64+31:0] expected[$],item,held;
  bit stalled=0;
  int sent=0,received=0,concurrent_cycles=0,blocked_writes=0;
  int unsigned rng=32'hbafc2564;
  tile_banks #(.P(P),.WORDS(WORDS)) dut(.*);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk)begin rng=next_rng(rng);m_ready=aresetn&&(rng[1]||rng[4]);end
  always @(posedge aclk)if(aresetn)begin
    if(stalled&&(!m_valid||{m_tag,m_data}!==held))$fatal(1,"bank output unstable");
    stalled=m_valid&&!m_ready;held={m_tag,m_data};
    if(load_valid&&!load_ready&&compute_active&&load_slot==compute_slot)blocked_writes++;
    if(load_valid&&load_ready)for(int b=0;b<8;b++)if(load_keep[b])model[load_slot][load_bank][load_addr][b*8+:8]=load_data[b*8+:8];
    if(s_valid&&s_ready)begin
      item[P*64+:32]=s_tag;
      for(int bank=0;bank<P;bank++)item[bank*64+:64]=model[compute_slot][bank][s_addr[bank*AW+:AW]];
      expected.push_back(item);sent++;
      if(load_valid&&load_ready)concurrent_cycles++;
    end
    if(m_valid&&m_ready)begin
      if(!expected.size())$fatal(1,"unexpected bank output");
      item=expected.pop_front();
      if({m_tag,m_data}!==item)$fatal(1,"bank mismatch P=%0d output=%0d",P,received);
      received++;
    end
  end
  task automatic load_word(input int slot,input int bank,input int address,input logic[63:0] data,input logic[7:0] keep);
    @(negedge aclk);load_valid=1;load_slot=slot;load_bank=bank;load_addr=address;load_data=data;load_keep=keep;
    do @(posedge aclk);while(!load_ready);
    @(negedge aclk);load_valid=0;
  endtask
  task automatic read_word(input int tag);
    @(negedge aclk);s_valid=1;s_tag=tag;
    for(int b=0;b<P;b++)begin rng=next_rng(rng);s_addr[b*AW+:AW]=rng%WORDS;end
    do @(posedge aclk);while(!s_ready);
    @(negedge aclk);s_valid=0;
  endtask
  initial begin
    repeat(4)@(negedge aclk);aresetn=1;
    for(int slot=0;slot<2;slot++)for(int b=0;b<P;b++)for(int a=0;a<WORDS;a++)
      load_word(slot,b,a,64'h1020304050607080^(slot<<20)^(b<<12)^a,8'hff);
    compute_active=1;compute_slot=0;
    fork
      begin for(int j=0;j<400;j++)read_word(j);end
      begin
        for(int j=0;j<150;j++)begin
          rng=next_rng(rng);load_word(1,j%P,j%WORDS,{rng,~rng},8'h55);
        end
      end
    join
    wait(received==sent);
    // Owner protection is independent of whether a read is being issued.
    @(negedge aclk);load_valid=1;load_slot=0;load_bank=0;load_addr=0;
    repeat(8)begin @(posedge aclk);if(load_ready)$fatal(1,"active slot accepted a write");end
    @(negedge aclk);load_valid=0;compute_slot=1;
    for(int j=0;j<100;j++)read_word(400+j);
    wait(received==sent);
    @(negedge aclk);s_valid=1;s_addr[0+:AW]=WORDS;
    @(posedge aclk);if(s_ready)$fatal(1,"out-of-range read accepted");
    @(negedge aclk);s_valid=0;
    if(!address_error)$fatal(1,"missing range error");
    aresetn=0;repeat(3)@(negedge aclk);aresetn=1;
    @(negedge aclk);if(address_error||m_valid)$fatal(1,"reset did not clear bank status");
    for(int j=0;j<20;j++)read_word(500+j);
    wait(received==sent);repeat(4)@(negedge aclk);
    if(concurrent_cycles==0||blocked_writes==0)$fatal(1,"bank concurrency coverage missing");
    $display("TB_PASS tile_banks P=%0d reads=%0d concurrent=%0d blocked_owner=%0d",P,received,concurrent_cycles,blocked_writes);$finish;
  end
  initial begin #1000000;$fatal(1,"bank watchdog");end
endmodule
module tb_tile_banks_p4;
  tb_tile_banks #(.P(4)) test();
endmodule
module tb_tile_banks_p6;
  tb_tile_banks #(.P(6)) test();
endmodule
