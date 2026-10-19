`timescale 1ns/1ps
module tb_frame_engine #(parameter integer P=4);
  logic aclk=0,aresetn=0,clear_cache=0,start=0;
  logic [31:0] cfg_program=32'h10800000,cfg_epoch=1;
  logic [15:0] cfg_count=0;
  logic [31:0] error_tag,cycles,input_hits,parameter_hits,overlap_cycles;
  logic busy,done,fault;logic [15:0] error,submitted,completed;
  logic [71:0] read_cmd_data,write_cmd_data;
  logic read_cmd_valid,read_cmd_ready,write_cmd_valid,write_cmd_ready;
  logic [63:0] read_data,write_data;
  logic [7:0] read_keep,read_status_data,write_status_data,write_keep;
  logic read_last,read_valid,read_status_valid,read_ready,read_status_ready;
  logic write_last,write_valid,write_ready,write_status_valid,write_status_ready;
  integer checked_words,expected_words,read_words,read_commands,write_commands;
  int fd,rc,njobs,wall_cycles=0;
  bit negative=0;
  frame_engine #(.P(P)) dut(.*);
  datamover_memory ddr(.*);
  always #5 aclk=~aclk;
  always @(posedge aclk)if(aresetn)begin
    wall_cycles++;
    if(fault&&!negative)$fatal(1,"frame fault P=%0d error=%h tag=%0d",P,error,error_tag);
    if((dut.executor.input_start||dut.executor.parameter_start||dut.executor.lut_start)&&dut.executor.ld[380])
      if(completed!=dut.executor.ld[415:384])$fatal(1,"DDR dispatch crossed layer barrier");
    if(wall_cycles>50000000)$fatal(1,"frame engine deadlock");
  end
  task automatic launch;
    @(negedge aclk);start=1;@(negedge aclk);start=0;
  endtask
  task automatic reset;
    @(negedge aclk);aresetn=0;repeat(4)@(negedge aclk);aresetn=1;
  endtask
  initial begin
    fd=$fopen("jobs.txt","r");if(!fd)$fatal(1,"missing job count");
    rc=$fscanf(fd,"%d\n",njobs);$fclose(fd);cfg_count=16'(njobs);
    reset();launch();wait(done);@(negedge aclk);
    if(fault||busy||checked_words!=expected_words||submitted!=njobs||completed!=njobs||input_hits==0||overlap_cycles==0)
      $fatal(1,"frame incomplete jobs=%0d/%0d/%0d writes=%0d/%0d",submitted,completed,njobs,checked_words,expected_words);
    $display("FRAME_RESULT P=%0d jobs=%0d writes=%0d reads=%0d commands=%0d/%0d cycles=%0d input_hits=%0d parameter_hits=%0d overlap=%0d",P,njobs,checked_words,read_words,read_commands,write_commands,cycles,input_hits,parameter_hits,overlap_cycles);
    negative=1;reset();cfg_count=0;launch();wait(done);@(negedge aclk);
    if(!fault||error!=16'h0701)$fatal(1,"zero-length execution program accepted");
    reset();cfg_count=1;cfg_program=32'h10800008;launch();wait(done);@(negedge aclk);
    if(!fault||error!=16'h0701)$fatal(1,"unaligned execution program accepted");
    reset();cfg_count=1;cfg_program=32'h10800000;cfg_epoch=2;launch();wait(done);@(negedge aclk);
    if(!fault||error!=16'h0702||submitted!=0)$fatal(1,"stale program epoch accepted");
    $display("TB_PASS frame_engine P=%0d conv=15 nodes=21 cpu_submissions=1 negative=3",P);$finish;
  end
endmodule
module tb_frame_engine_p4;tb_frame_engine #(.P(4)) test();endmodule
module tb_frame_engine_p6;tb_frame_engine #(.P(6)) test();endmodule
