`timescale 1ns/1ps
module tb_dma_transfer;
  logic aclk=0,aresetn=0;
  logic [31:0] req_address=0;
  logic [22:0] req_bytes=0;
  logic [3:0] req_tag=0;
  logic req_valid=0,req_ready;
  logic [71:0] cmd_data;
  logic cmd_valid,cmd_ready=0;
  logic [7:0] status_data=0;
  logic status_valid=0,status_ready;
  logic [63:0] s_data=0,m_data;
  logic [7:0] s_keep=8'hff,m_keep;
  logic s_last=0,s_valid=0,s_ready,m_last,m_valid,m_ready=0;
  logic busy,done,fault;
  logic [7:0] error;
  logic [31:0] actual_bytes;
  int unsigned rng=32'hadcd0225;
  int commands=0,successful=0,negative=0;
  bit block_command=0,stalled=0;
  logic [72:0] held;
  dma_transfer #(.TIMEOUT_CYCLES(512)) dut(.*);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk)begin
    rng=next_rng(rng);cmd_ready=aresetn&&!block_command&&(rng[0]||rng[3]);
    m_ready=aresetn&&(rng[1]||rng[5]);
  end
  always @(posedge aclk)begin
    if(!aresetn)stalled=0;
    else begin
      if(cmd_valid&&cmd_ready)begin
        if(cmd_data[71:68]!=0||cmd_data[67:64]!=req_tag||cmd_data[63:32]!=req_address||
           cmd_data[31:24]!=8'h40||!cmd_data[23]||cmd_data[22:0]!=req_bytes)
          $fatal(1,"DataMover command format mismatch %h",cmd_data);
        commands++;
      end
      if(stalled&&(!m_valid||{m_last,m_keep,m_data}!==held))$fatal(1,"DMA stream unstable under stall");
      stalled=m_valid&&!m_ready;held={m_last,m_keep,m_data};
    end
  end
  task automatic reset_monitor;
    @(negedge aclk);aresetn=0;req_valid=0;s_valid=0;status_valid=0;block_command=0;
    repeat(3)@(negedge aclk);aresetn=1;
    @(negedge aclk);if(fault||busy||error||!req_ready)$fatal(1,"DMA reset failed");
  endtask
  task automatic request(input int bytes,input int tag,input logic[31:0] address);
    @(negedge aclk);req_bytes=bytes;req_address=address;req_tag=tag;req_valid=1;
    do @(posedge aclk);while(!req_ready);
    @(negedge aclk);req_valid=0;
  endtask
  task automatic data_packet(input int count,input bit last_enabled,input bit bad_keep);
    for(int j=0;j<count;j++)begin
      @(negedge aclk);s_valid=0;
      repeat(j%3)@(negedge aclk);
      s_data=64'hacdcabcd10203040^64'(j);s_keep=(bad_keep&&j==0)?8'h7f:8'hff;
      s_last=last_enabled&&j==count-1;s_valid=1;
      do @(posedge aclk);while(!s_ready);
      if(m_data!==s_data||m_keep!==s_keep||m_last!==s_last)$fatal(1,"DMA payload corrupted");
    end
    @(negedge aclk);s_valid=0;s_last=0;
  endtask
  task automatic status_packet(input logic[7:0] value,input int delay_cycles);
    repeat(delay_cycles)@(negedge aclk);
    @(negedge aclk);status_data=value;status_valid=1;
    do @(posedge aclk);while(!status_ready);
    @(negedge aclk);status_valid=0;
  endtask
  task automatic run(input int mode,input int count);
    int words;
    logic[7:0] response;
    reset_monitor();
    request(count*8,mode,32'h12000000+mode*4096);
    do @(posedge aclk);while(!(cmd_valid&&cmd_ready));
    words=mode==3?count-1:mode==9?count+1:count;
    response=8'h80|req_tag;
    if(mode==1)response|=8'h40;
    if(mode==2)response^=8'h01;
    fork
      data_packet(words,mode!=8,mode==4);
      begin if(mode!=7)status_packet(response,mode==5?0:count*4);end
    join
    wait(done);@(negedge aclk);
    if(mode==0||mode==5)begin
      if(fault||error||actual_bytes!=count*8)$fatal(1,"normal DMA failed mode=%0d err=%h",mode,error);
      successful++;
    end else begin
      if(!fault||!error||req_ready)$fatal(1,"bad transfer accepted mode=%0d error=%h",mode,error);
      if((mode==1||mode==2)&&!error[2])$fatal(1,"missing status validation");
      if((mode==3||mode==4||mode==9)&&!error[1])$fatal(1,"missing data validation");
      if((mode==7||mode==8)&&!error[3])$fatal(1,"missing watchdog");
      negative++;
    end
  endtask
  initial begin
    for(int j=1;j<=32;j++)run(0,j);
    run(5,17);
    run(1,8);run(2,8);run(3,8);run(4,8);run(7,8);run(8,8);run(9,8);
    for(int mode=0;mode<4;mode++)begin
      int before_commands;
      reset_monitor();before_commands=commands;
      case(mode)
        0:request(0,0,32'h10000000);
        1:request(8,0,32'h10000001);
        2:request(7,0,32'h10000000);
        3:request(16,0,32'hfffffff8);
      endcase
      wait(done);@(negedge aclk);
      if(!fault||error!=1||commands!=before_commands)$fatal(1,"malformed request reached DMA");
      negative++;
    end
    reset_monitor();block_command=1;request(64,0,32'h10000000);
    wait(done);@(negedge aclk);
    if(!fault||!error[3])$fatal(1,"command channel watchdog missing");negative++;
    run(0,8);
    $display("TB_PASS dma_transfer success=%0d expected_faults=%0d commands=%0d",successful,negative,commands);$finish;
  end
  initial begin #2000000;$fatal(1,"DMA TB watchdog");end
endmodule
