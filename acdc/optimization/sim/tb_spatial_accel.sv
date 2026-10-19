`timescale 1ns/1ps
module tb_spatial_accel #(parameter integer P=4);
  logic aclk=0,aresetn=0,interrupt;
  logic [7:0] s_axi_awaddr=0,s_axi_araddr=0;
  logic s_axi_awvalid=0,s_axi_awready,s_axi_wvalid=0,s_axi_wready,s_axi_bvalid,s_axi_bready=0;
  logic s_axi_arvalid=0,s_axi_arready,s_axi_rvalid,s_axi_rready=0;
  logic [31:0] s_axi_wdata=0,s_axi_rdata;logic [3:0] s_axi_wstrb=15;
  logic [1:0] s_axi_bresp,s_axi_rresp;
  logic [71:0] read_cmd_data,write_cmd_data;
  logic read_cmd_valid,read_cmd_ready,write_cmd_valid,write_cmd_ready;
  logic [63:0] read_data,write_data;
  logic [7:0] read_keep,read_status_data,write_status_data,write_keep;
  logic read_last,read_valid,read_status_valid,read_ready,read_status_ready;
  logic write_last,write_valid,write_ready,write_status_valid,write_status_ready;
  integer checked_words,expected_words,read_words,read_commands,write_commands;
  int fd,rc,njobs,wall_cycles=0;logic [31:0] value;
  spatial_accel #(.P(P)) dut(.m_axis_read_cmd_tdata(read_cmd_data),.m_axis_read_cmd_tvalid(read_cmd_valid),.m_axis_read_cmd_tready(read_cmd_ready),
    .s_axis_read_data_tdata(read_data),.s_axis_read_data_tkeep(read_keep),.s_axis_read_data_tlast(read_last),
    .s_axis_read_data_tvalid(read_valid),.s_axis_read_data_tready(read_ready),
    .s_axis_read_status_tdata(read_status_data),.s_axis_read_status_tvalid(read_status_valid),.s_axis_read_status_tready(read_status_ready),
    .m_axis_write_cmd_tdata(write_cmd_data),.m_axis_write_cmd_tvalid(write_cmd_valid),.m_axis_write_cmd_tready(write_cmd_ready),
    .m_axis_write_data_tdata(write_data),.m_axis_write_data_tkeep(write_keep),.m_axis_write_data_tlast(write_last),
    .m_axis_write_data_tvalid(write_valid),.m_axis_write_data_tready(write_ready),
    .s_axis_write_status_tdata(write_status_data),.s_axis_write_status_tvalid(write_status_valid),.s_axis_write_status_tready(write_status_ready),.*);
  datamover_memory ddr(.*);
  always #5 aclk=~aclk;
  always @(posedge aclk)if(aresetn)begin
    wall_cycles++;
    if(dut.engine.fault)$fatal(1,"AXI-controlled frame fault error=%h tag=%0d",dut.engine.error,dut.engine.error_tag);
    if(wall_cycles>50000000)$fatal(1,"AXI-controlled frame deadlock");
  end
  task automatic write_reg(input logic [7:0] address,input logic [31:0] data);
    fork
      begin
        @(negedge aclk);s_axi_awaddr=address;s_axi_awvalid=1;
        do @(posedge aclk);while(!s_axi_awready);
        @(negedge aclk);s_axi_awvalid=0;
      end
      begin
        repeat(3)@(negedge aclk);s_axi_wdata=data;s_axi_wvalid=1;
        do @(posedge aclk);while(!s_axi_wready);
        @(negedge aclk);s_axi_wvalid=0;
      end
    join
    wait(s_axi_bvalid);if(s_axi_bresp!=0)$fatal(1,"AXI configuration rejected");
    @(negedge aclk);s_axi_bready=1;@(negedge aclk);s_axi_bready=0;
  endtask
  task automatic read_reg(input logic [7:0] address);
    @(negedge aclk);s_axi_araddr=address;s_axi_arvalid=1;
    do @(posedge aclk);while(!s_axi_arready);
    @(negedge aclk);s_axi_arvalid=0;
    wait(s_axi_rvalid);value=s_axi_rdata;if(s_axi_rresp!=0)$fatal(1,"AXI result read rejected");
    s_axi_rready=1;@(negedge aclk);s_axi_rready=0;
  endtask
  initial begin
    fd=$fopen("jobs.txt","r");if(!fd)$fatal(1,"missing job count");rc=$fscanf(fd,"%d\n",njobs);$fclose(fd);
    repeat(25)@(negedge aclk);aresetn=1;
    read_reg(0);if(value!=32'h41434438)$fatal(1,"wrong hardware ABI");
    read_reg(8);if(value[7:0]!=P)$fatal(1,"wrong hardware profile");
    write_reg(8'h14,32'h10800000);write_reg(8'h18,njobs);write_reg(8'h1c,1);write_reg(8'h0c,1);
    wait(interrupt);
    read_reg(8'h10);if(value[2:0]!=3'b010)$fatal(1,"wrong completed frame status %h",value);
    read_reg(8'h2c);if(value!={16'(njobs),16'(njobs)})$fatal(1,"missing completed jobs");
    read_reg(8'h28);
    if(checked_words!=expected_words)$fatal(1,"AXI-controlled frame output incomplete");
    $display("TB_PASS spatial_accel P=%0d ABI=2.0 conv=15 nodes=21 jobs=%0d writes=%0d cycles=%0d",P,njobs,checked_words,value);$finish;
  end
endmodule
module tb_spatial_accel_p4;tb_spatial_accel #(.P(4)) test();endmodule
module tb_spatial_accel_p6;tb_spatial_accel #(.P(6)) test();endmodule
