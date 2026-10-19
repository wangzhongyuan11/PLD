`timescale 1ns/1ps
module tb_tile_executor #(parameter integer P=4);
  logic aclk=0,aresetn=0,clear_cache=0,s_valid=0,s_ready;
  logic [511:0] s_descriptor=0;
  logic [31:0] completion_tag,error_tag,input_hits,parameter_hits,overlap_cycles;
  logic completion_valid,completion_ready=0,busy,fault;logic [15:0] error;
  logic [71:0] read_cmd_data,write_cmd_data;
  logic read_cmd_valid,read_cmd_ready,write_cmd_valid,write_cmd_ready;
  logic [63:0] read_data,write_data;
  logic [7:0] read_keep,read_status_data,write_status_data,write_keep;
  logic read_last,read_valid,read_status_valid,read_ready,read_status_ready;
  logic write_last,write_valid,write_ready,write_status_valid,write_status_ready;
  integer checked_words,expected_words,read_words,read_commands,write_commands;
  int fd,rc,njobs,completed=0,cycles=0,barriers=0;
  int unsigned rng=32'hacdc9988;
  tile_executor #(.P(P)) dut(.*);
  datamover_memory ddr(.*);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk)begin rng=next_rng(rng);completion_ready=aresetn&&(rng[0]||rng[5]);end
  always @(posedge aclk)if(aresetn)begin
    cycles++;
    if(fault)$fatal(1,"executor fault P=%0d error=%h tag=%0d",P,error,error_tag);
    if(completion_valid&&completion_ready)begin
      if(completion_tag!=completed)$fatal(1,"executor unordered completion");completed++;
    end
    if((dut.input_start||dut.parameter_start||dut.lut_start)&&dut.ld[380])begin
      if(completed!=dut.ld[415:384])$fatal(1,"layer barrier allowed prefetch before prior checked completion");
      if(dut.input_start)barriers++;
    end
    if(cycles>50000000)$fatal(1,"executor deadlock");
  end
  initial begin
    fd=$fopen("jobs.txt","r");if(!fd)$fatal(1,"missing job program");
    rc=$fscanf(fd,"%d\n",njobs);
    repeat(5)@(negedge aclk);aresetn=1;
    for(int j=0;j<njobs;j++)begin
      @(negedge aclk);rc=$fscanf(fd,"%h\n",s_descriptor);if(rc!=1)$fatal(1,"short job program");s_valid=1;
      do @(posedge aclk);while(!s_ready);
      @(negedge aclk);s_valid=0;
    end
    wait(completed==njobs);repeat(5)@(negedge aclk);
    if(busy||checked_words!=expected_words||input_hits==0||overlap_cycles==0||barriers!=15)
      $fatal(1,"executor incomplete busy=%b writes=%0d/%0d hits=%0d overlap=%0d barriers=%0d",busy,checked_words,expected_words,input_hits,overlap_cycles,barriers);
    $display("TB_PASS tile_executor P=%0d jobs=%0d conv=15 nodes=21 writes=%0d reads=%0d commands=%0d/%0d cycles=%0d input_hits=%0d parameter_hits=%0d overlap=%0d",P,njobs,checked_words,read_words,read_commands,write_commands,cycles,input_hits,parameter_hits,overlap_cycles);$finish;
  end
endmodule
module tb_tile_executor_p4;tb_tile_executor #(.P(4)) test();endmodule
module tb_tile_executor_p6;tb_tile_executor #(.P(6)) test();endmodule
