`timescale 1ns/1ps
// B command/data/status monitors -> actual generated AMD DataMover -> AXI RAM.
// RAM is a single-outstanding, stalled AXI slave, not a DataMover replacement.
module tb_vendor_datamover;
  logic aclk=0,aresetn,halt;
  logic [31:0] address;
  logic [22:0] bytes;
  logic [3:0] tag;
  logic read_request,write_request,read_request_ready,write_request_ready;
  wire [71:0] read_cmd,write_cmd;
  wire read_cv,read_cr,write_cv,write_cr;
  wire [7:0] read_status,write_status;
  wire read_sv,read_sr,write_sv,write_sr;
  wire [63:0] ip_read_data,checked_read_data,ip_write_data;
  wire [7:0] ip_read_keep,checked_read_keep,ip_write_keep;
  wire ip_read_last,ip_read_valid,ip_read_ready;
  wire checked_read_last,checked_read_valid;
  logic checked_read_ready;
  wire ip_write_last,ip_write_valid,ip_write_ready;
  logic [63:0] source_data;
  logic [7:0] source_keep;
  logic source_last,source_valid;
  wire source_ready,read_done,write_done,read_fault,write_fault;
  wire [7:0] read_error,write_error;
  wire [31:0] read_actual,write_actual;
  wire mm2s_halted,s2mm_halted,mm2s_error,s2mm_error;
  wire [31:0] araddr,awaddr;
  wire [7:0] arlen,awlen;
  wire [2:0] arsize,awsize;
  wire [1:0] arburst,awburst;
  wire arvalid,awvalid,rready,wvalid,wlast,bready;
  logic arready,awready,wready,rvalid,rlast,bvalid;
  logic [63:0] rdata;
  wire [63:0] wdata;
  wire [7:0] wstrb;
  logic [1:0] rresp,bresp;
  logic [63:0] memory[0:4095];
  logic read_active,write_active;
  logic [31:0] read_address,write_address;
  integer read_left,write_left,response_delay;
  logic read_bad,write_bad,block_read;
  logic read_completed,write_completed;
  integer read_seen=0,read_bursts=0,write_bursts=0,read_stalls=0,write_stalls=0,successes=0;
  int unsigned rng=32'h71825;
  dma_transfer #(.TIMEOUT_CYCLES(4096)) read_monitor(
    .aclk,.aresetn,.req_address(address),.req_bytes(bytes),.req_tag(tag),
    .req_valid(read_request),.req_ready(read_request_ready),
    .cmd_data(read_cmd),.cmd_valid(read_cv),.cmd_ready(read_cr),
    .status_data(read_status),.status_valid(read_sv),.status_ready(read_sr),
    .s_data(ip_read_data),.s_keep(ip_read_keep),.s_last(ip_read_last),
    .s_valid(ip_read_valid),.s_ready(ip_read_ready),
    .m_data(checked_read_data),.m_keep(checked_read_keep),.m_last(checked_read_last),
    .m_valid(checked_read_valid),.m_ready(checked_read_ready),
    .busy(),.done(read_done),.fault(read_fault),.error(read_error),.actual_bytes(read_actual));
  dma_transfer #(.TIMEOUT_CYCLES(4096)) write_monitor(
    .aclk,.aresetn,.req_address(address),.req_bytes(bytes),.req_tag(tag),
    .req_valid(write_request),.req_ready(write_request_ready),
    .cmd_data(write_cmd),.cmd_valid(write_cv),.cmd_ready(write_cr),
    .status_data(write_status),.status_valid(write_sv),.status_ready(write_sr),
    .s_data(source_data),.s_keep(source_keep),.s_last(source_last),
    .s_valid(source_valid),.s_ready(source_ready),
    .m_data(ip_write_data),.m_keep(ip_write_keep),.m_last(ip_write_last),
    .m_valid(ip_write_valid),.m_ready(ip_write_ready),
    .busy(),.done(write_done),.fault(write_fault),.error(write_error),.actual_bytes(write_actual));
  dma_top_datamover_0 ip(
    .m_axi_mm2s_aclk(aclk),.m_axi_mm2s_aresetn(aresetn),
    .m_axis_mm2s_cmdsts_aclk(aclk),.m_axis_mm2s_cmdsts_aresetn(aresetn),
    .mm2s_halt(halt),.mm2s_halt_cmplt(mm2s_halted),.mm2s_err(mm2s_error),
    .mm2s_allow_addr_req(1'b1),.mm2s_dbg_sel(4'd0),
    .s_axis_mm2s_cmd_tdata(read_cmd),.s_axis_mm2s_cmd_tvalid(read_cv),.s_axis_mm2s_cmd_tready(read_cr),
    .m_axis_mm2s_sts_tdata(read_status),.m_axis_mm2s_sts_tvalid(read_sv),.m_axis_mm2s_sts_tready(read_sr),
    .m_axis_mm2s_tdata(ip_read_data),.m_axis_mm2s_tkeep(ip_read_keep),.m_axis_mm2s_tlast(ip_read_last),
    .m_axis_mm2s_tvalid(ip_read_valid),.m_axis_mm2s_tready(ip_read_ready),
    .m_axi_mm2s_araddr(araddr),.m_axi_mm2s_arlen(arlen),.m_axi_mm2s_arsize(arsize),
    .m_axi_mm2s_arburst(arburst),.m_axi_mm2s_arvalid(arvalid),.m_axi_mm2s_arready(arready),
    .m_axi_mm2s_rdata(rdata),.m_axi_mm2s_rresp(rresp),.m_axi_mm2s_rlast(rlast),
    .m_axi_mm2s_rvalid(rvalid),.m_axi_mm2s_rready(rready),
    .m_axi_s2mm_aclk(aclk),.m_axi_s2mm_aresetn(aresetn),
    .m_axis_s2mm_cmdsts_awclk(aclk),.m_axis_s2mm_cmdsts_aresetn(aresetn),
    .s2mm_halt(halt),.s2mm_halt_cmplt(s2mm_halted),.s2mm_err(s2mm_error),
    .s2mm_allow_addr_req(1'b1),.s2mm_dbg_sel(4'd0),
    .s_axis_s2mm_cmd_tdata(write_cmd),.s_axis_s2mm_cmd_tvalid(write_cv),.s_axis_s2mm_cmd_tready(write_cr),
    .m_axis_s2mm_sts_tdata(write_status),.m_axis_s2mm_sts_tvalid(write_sv),.m_axis_s2mm_sts_tready(write_sr),
    .s_axis_s2mm_tdata(ip_write_data),.s_axis_s2mm_tkeep(ip_write_keep),.s_axis_s2mm_tlast(ip_write_last),
    .s_axis_s2mm_tvalid(ip_write_valid),.s_axis_s2mm_tready(ip_write_ready),
    .m_axi_s2mm_awaddr(awaddr),.m_axi_s2mm_awlen(awlen),.m_axi_s2mm_awsize(awsize),
    .m_axi_s2mm_awburst(awburst),.m_axi_s2mm_awvalid(awvalid),.m_axi_s2mm_awready(awready),
    .m_axi_s2mm_wdata(wdata),.m_axi_s2mm_wstrb(wstrb),.m_axi_s2mm_wlast(wlast),
    .m_axi_s2mm_wvalid(wvalid),.m_axi_s2mm_wready(wready),
    .m_axi_s2mm_bresp(bresp),.m_axi_s2mm_bvalid(bvalid),.m_axi_s2mm_bready(bready));
  always #5 aclk=~aclk;
  function automatic int unsigned random_next(input int unsigned value);
    value^=value<<13;value^=value>>17;value^=value<<5;return value;
  endfunction
  function automatic logic [63:0] pattern(input int unsigned i);
    return 64'hf192acdc76543210^(64'(i)*64'h01010107100b0001);
  endfunction
  always @(negedge aclk)begin
    rng=random_next(rng);
    arready=aresetn&&!block_read&&!read_active&&!rvalid&&(rng[0]||rng[3]);
    awready=aresetn&&!write_active&&!bvalid&&response_delay==0&&(rng[1]||rng[4]);
    wready=aresetn&&write_active&&(rng[2]||rng[7]);
    checked_read_ready=aresetn&&(rng[6]||rng[9]);
  end
  always @(posedge aclk)begin
    if(!aresetn)begin
      read_active<=0;write_active<=0;rvalid<=0;rlast<=0;bvalid<=0;response_delay<=0;
      read_completed<=0;write_completed<=0;read_seen<=0;
    end else begin
      if(read_done)read_completed<=1;
      if(write_done)write_completed<=1;
      if(arvalid&&!arready)read_stalls++;
      if(wvalid&&!wready)write_stalls++;
      if(arvalid&&arready)begin
        if(arsize!=3||arburst!=1||arlen>15||araddr[2:0]!=0||
           {1'b0,araddr[11:0]}+(int'(arlen)+1)*8>4096||araddr+(int'(arlen)+1)*8>32768)
          $fatal(1,"vendor MM2S AXI burst invalid address=%h len=%0d",araddr,arlen);
        read_active<=1;read_address<=araddr;read_left<=int'(arlen)+1;read_bursts++;
      end
      if(rvalid&&rready)rvalid<=0;
      if(read_active&&(!rvalid||rready)&&(rng[10]||rng[12]))begin
        rvalid<=1;rdata<=memory[read_address/8];rresp<=read_bad?2'b10:2'b00;rlast<=read_left==1;
        read_address<=read_address+8;read_left<=read_left-1;
        if(read_left==1)read_active<=0;
      end
      if(awvalid&&awready)begin
        if(awsize!=3||awburst!=1||awlen>15||awaddr[2:0]!=0||
           {1'b0,awaddr[11:0]}+(int'(awlen)+1)*8>4096||awaddr+(int'(awlen)+1)*8>32768)
          $fatal(1,"vendor S2MM AXI burst invalid address=%h len=%0d",awaddr,awlen);
        write_active<=1;write_address<=awaddr;write_left<=int'(awlen)+1;write_bursts++;
      end
      if(wvalid&&wready)begin
        if(wstrb!==8'hff||wlast!==(write_left==1))$fatal(1,"vendor WSTRB/WLAST mismatch");
        memory[write_address/8]<=wdata;write_address<=write_address+8;write_left<=write_left-1;
        if(write_left==1)begin write_active<=0;response_delay<=4;end
      end
      if(bvalid&&bready)bvalid<=0;
      if(response_delay>0)begin
        response_delay<=response_delay-1;
        if(response_delay==1)begin bvalid<=1;bresp<=write_bad?2'b10:2'b00;end
      end
      if(checked_read_valid&&checked_read_ready)begin
        if(checked_read_keep!==8'hff||checked_read_last!==(read_seen+1==bytes/8)||
           checked_read_data!==memory[address/8+read_seen])$fatal(1,"MM2S checked payload mismatch word=%0d",read_seen);
        read_seen<=read_seen+1;
      end
    end
  end
  task automatic reset_all;
    @(negedge aclk);aresetn=0;read_request=0;write_request=0;source_valid=0;halt=0;
    repeat(20)@(negedge aclk);aresetn=1;repeat(20)@(negedge aclk);
    if(mm2s_error||s2mm_error||read_fault||write_fault)$fatal(1,"vendor/common reset failed");
  endtask
  task automatic request(input bit write_mode);
    @(negedge aclk);if(write_mode)write_request=1;else read_request=1;
    do @(posedge aclk);while(write_mode?!write_request_ready:!read_request_ready);
    @(negedge aclk);read_request=0;write_request=0;
  endtask
  task automatic send_data;
    for(int i=0;i<bytes/8;i++)begin
      @(negedge aclk);source_valid=0;repeat(i%3)@(negedge aclk);
      source_data=pattern(i+tag*4096);source_last=i+1==bytes/8;source_valid=1;
      do @(posedge aclk);while(!source_ready);
    end
    @(negedge aclk);source_valid=0;source_last=0;
  endtask
  task automatic pair(input int start_address,input int count,input int request_tag);
    read_bad=0;write_bad=0;block_read=0;reset_all();
    address=start_address;bytes=count;tag=4'(request_tag);
    request(1);send_data();wait(write_completed);@(negedge aclk);
    if(write_fault||write_error||write_actual!=bytes||s2mm_error)$fatal(1,"S2MM transfer/status fault");
    for(int i=0;i<count/8;i++)if(memory[start_address/8+i]!==pattern(i+tag*4096))$fatal(1,"AXI stored bytes differ");
    request(0);wait(read_completed);@(negedge aclk);
    if(read_fault||read_error||read_actual!=bytes||read_seen!=bytes/8||mm2s_error)$fatal(1,"MM2S transfer/status fault");
    successes++;$display("VENDOR_PAIR address=%h bytes=%0d tag=%0d exact",address,bytes,tag);
  endtask
  initial begin
    #1;aresetn=0;halt=0;read_request=0;write_request=0;address=0;bytes=8;tag=0;
    source_data=0;source_keep=8'hff;source_last=0;source_valid=0;
    read_bad=0;write_bad=0;block_read=0;
    for(int i=0;i<4096;i++)memory[i]=64'hdeadbeeffeed8000^64'(i);
    repeat(25)@(negedge aclk);
    pair(0,8,0);pair(512,72,3);pair(4088,200,7);pair(4000,8192,15);
    // Real SLVERR responses must produce an IP status/monitor fault.
    reset_all();address=2048;bytes=128;tag=9;write_bad=1;
    request(1);send_data();wait(write_completed);@(negedge aclk);
    if(!write_fault||!write_error[2])$fatal(1,"S2MM AXI error accepted");
    write_bad=0;reset_all();read_bad=1;
    request(0);wait(read_completed);@(negedge aclk);
    if(!read_fault||!read_error[2])$fatal(1,"MM2S AXI error accepted");
    read_bad=0;reset_all();block_read=1;
    request(0);wait(read_completed);@(negedge aclk);
    if(!read_fault||!read_error[3])$fatal(1,"stalled vendor read did not time out");
    block_read=0;
    // Reset a live multi-burst transfer, then require a fresh correct transaction.
    reset_all();bytes=8192;address=4096;tag=5;
    request(0);wait(read_seen>=3);reset_all();pair(16384,256,6);
    @(negedge aclk);halt=1;wait(mm2s_halted&&s2mm_halted);
    if(!read_stalls||!write_stalls)$fatal(1,"missing randomized backpressure coverage");
    $display("TB_PASS vendor_datamover pairs=%0d read_bursts=%0d write_bursts=%0d read_stalls=%0d write_stalls=%0d AXI_errors=2 timeout=1 active_reset=1 halt=1",successes,read_bursts,write_bursts,read_stalls,write_stalls);
    $finish;
  end
  initial begin #5000000;$fatal(1,"vendor DataMover TB watchdog");end
endmodule
