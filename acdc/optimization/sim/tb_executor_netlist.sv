`timescale 1ns/1ps
// Same evolving DDR oracle as RTL simulation, no access to internal RTL state.
module tb_executor_netlist;
  logic aclk=0,aresetn,clear_cache,s_valid,s_ready;
  logic [511:0] s_descriptor;
  logic [31:0] completion_tag,error_tag,input_hits,parameter_hits,overlap_cycles;
  logic completion_valid,completion_ready,busy,fault;logic [15:0] error;
  logic [71:0] read_cmd_data,write_cmd_data;
  logic read_cmd_valid,read_cmd_ready,write_cmd_valid,write_cmd_ready;
  logic [63:0] read_data,write_data;
  logic [7:0] read_keep,read_status_data,write_status_data,write_keep;
  logic read_last,read_valid,read_status_valid,read_ready,read_status_ready;
  logic write_last,write_valid,write_ready,write_status_valid,write_status_ready;
  integer checked_words,expected_words,read_words,read_commands,write_commands;
  int fd,rc,njobs,completed=0,cycles=0,core_beats=0;
  tile_executor dut(.*);
`ifdef ACDC_PORTABLE_DDR
  datamover_memory_portable ddr(.*);
`else
  datamover_memory ddr(.*);
`endif
  always #5 aclk=~aclk;
  always @(posedge aclk)if(aresetn)begin
    cycles++;
    if(fault)$fatal(1,"netlist fault error=%h tag=%0d",error,error_tag);
    if(cycles==10000&&checked_words==0)$fatal(1,"netlist no first write: read_commands=%0d read_words=%0d write_commands=%0d ready=%b busy=%b fault=%b error=%h completed=%0d",read_commands,read_words,write_commands,s_ready,busy,fault,error,completed);
    if(read_cmd_valid&&read_cmd_ready&&read_commands<20)$display("NETLIST_READ cycle=%0d address=%h bytes=%0d",cycles,read_cmd_data[63:32],read_cmd_data[22:0]);
    // Check only public pins: synthesis may legally flatten writer internals.
    // The DDR oracle checks every byte/address and completion tag below.
    if(completion_valid&&completion_ready)begin
      if(completion_tag!=completed)$fatal(1,"netlist unordered completion");completed++;
    end
    if(cycles>2000000)$fatal(1,"netlist deadlock");
  end
  initial begin
    // UNISIM LUT models use explicit input event sensitivity. In Icarus,
    // declaration-initialized registers can be collapsed onto primitive ports
    // before those processes wait for events, leaving a constant input cone X.
    // Drive every static input after time zero, while synchronous reset is held.
    // This changes stimulus scheduling only; no force or primitive substitution.
    #1;aresetn=0;clear_cache=0;s_valid=0;s_descriptor=0;completion_ready=1;
    fd=$fopen("jobs.txt","r");if(!fd)$fatal(1,"missing jobs");rc=$fscanf(fd,"%d\n",njobs);
    // Let the vendor primitive GSR (100ns) settle before releasing reset.
    repeat(25)@(negedge aclk);aresetn=1;
    for(int j=0;j<njobs;j++)begin
      @(negedge aclk);rc=$fscanf(fd,"%h\n",s_descriptor);if(rc!=1)$fatal(1,"short job program");s_valid=1;
      do @(posedge aclk);while(!s_ready);
      @(negedge aclk);s_valid=0;
    end
    wait(completed==njobs);repeat(5)@(negedge aclk);
    if(busy||checked_words!=expected_words)$fatal(1,"netlist incomplete");
    $display("TB_PASS executor_functional_netlist jobs=%0d writes=%0d cycles=%0d",njobs,checked_words,cycles);$finish;
  end
endmodule
