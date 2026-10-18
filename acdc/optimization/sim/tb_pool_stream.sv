`timescale 1ns/1ps
module tb_pool_stream;
  logic aclk=0,aresetn=0,start=0;
  logic [15:0] cfg_width=0,cfg_height=0;
  logic [63:0] source_data=0,raw_data,pool_input,pool_data;
  logic source_valid=0,source_ready,enabled=0;
  logic raw_valid,raw_ready=0,pool_input_valid,pool_input_ready;
  logic [7:0] pool_x,pool_y;
  logic pool_valid,pool_ready=0,busy,done,error;
  logic [63:0] raw_expected[8192],raw_held;
  logic [79:0] pool_expected[2048],pool_held;
  int fd,rf,pf,ncases,nraw,npool,rc,raw_count=0,pool_count=0,target_raw=0,cycles=0;
  int enabled_value,h,w,blocked=0;
  bit raw_stall=0,pool_stall=0;
  int unsigned rng=32'hacdc1202;
  stream_fork fork_pixels(.aclk,.aresetn,.s_data(source_data),.s_second_enable(enabled),
    .s_valid(source_valid),.s_ready(source_ready),.m0_data(raw_data),.m1_data(pool_input),
    .m0_valid(raw_valid),.m1_valid(pool_input_valid),.m0_ready(raw_ready),.m1_ready(pool_input_ready));
  pool_stream dut(.aclk,.aresetn,.start,.cfg_width,.cfg_height,
    .s_data(pool_input),.s_valid(pool_input_valid),.s_ready(pool_input_ready),
    .m_data(pool_data),.m_x(pool_x),.m_y(pool_y),.m_valid(pool_valid),.m_ready(pool_ready),.busy,.done,.error);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk)begin
    rng=next_rng(rng);raw_ready=aresetn&&(rng[0]||rng[4]);
    pool_ready=aresetn&&cycles%400>160&&(rng[1]||rng[5]);
  end
  always @(posedge aclk)if(aresetn)begin
    cycles++;if(source_valid&&!source_ready)blocked++;
    if(raw_stall&&(!raw_valid||raw_data!==raw_held))$fatal(1,"fork raw changed under stall");
    if(pool_stall&&(!pool_valid||{pool_y,pool_x,pool_data}!==pool_held))$fatal(1,"pool changed under stall");
    raw_stall=raw_valid&&!raw_ready;raw_held=raw_data;
    pool_stall=pool_valid&&!pool_ready;pool_held={pool_y,pool_x,pool_data};
    if(raw_valid&&raw_ready)begin
      if(raw_count>=nraw||raw_data!==raw_expected[raw_count])$fatal(1,"fork output duplicate/mismatch %0d",raw_count);
      raw_count++;
    end
    if(pool_valid&&pool_ready)begin
      if(pool_count>=npool||{pool_y,pool_x,pool_data}!==pool_expected[pool_count])$fatal(1,"signed pool mismatch %0d",pool_count);
      pool_count++;
    end
  end
  initial begin
    fd=$fopen("pool_input.txt","r");rf=$fopen("pool_raw_expected.txt","r");pf=$fopen("pool_expected.txt","r");
    if(!fd||!rf||!pf)$fatal(1,"missing pool vectors");
    rc=$fscanf(fd,"%d\n",ncases);rc=$fscanf(rf,"%d\n",nraw);rc=$fscanf(pf,"%d\n",npool);
    if(nraw>8192||npool>2048)$fatal(1,"pool vector capacity");
    for(int j=0;j<nraw;j++)rc=$fscanf(rf,"%h\n",raw_expected[j]);
    for(int j=0;j<npool;j++)rc=$fscanf(pf,"%h\n",pool_expected[j]);
    repeat(4)@(negedge aclk);aresetn=1;
    for(int c=0;c<ncases;c++)begin
      @(negedge aclk);rc=$fscanf(fd,"%h %h %h\n",enabled_value,h,w);
      if(rc!=3)$fatal(1,"bad pool case");enabled=enabled_value;cfg_height=h;cfg_width=w;
      start=enabled;@(negedge aclk);start=0;
      for(int j=0;j<h*w;j++)begin
        @(negedge aclk);rc=$fscanf(fd,"%h\n",source_data);source_valid=1;
        if(rc!=1)$fatal(1,"short pool input");
        do @(posedge aclk);while(!source_ready);
      end
      @(negedge aclk);source_valid=0;target_raw+=h*w;
      wait(raw_count==target_raw);
      if(enabled)begin wait(!busy);if(error)$fatal(1,"pool error");end
    end
    repeat(10)@(negedge aclk);
    if(pool_count!=npool||blocked==0)$fatal(1,"missing pool output/backpressure coverage");
    cfg_width=3;cfg_height=4;start=1;@(negedge aclk);start=0;
    if(!error||!done||busy)$fatal(1,"odd pool shape not rejected");
    aresetn=0;repeat(3)@(negedge aclk);aresetn=1;@(negedge aclk);
    if(error||pool_valid||raw_valid)$fatal(1,"pool/fork reset failed");
    $display("TB_PASS pool_stream cases=%0d raw=%0d pooled=%0d blocked=%0d",ncases,raw_count,pool_count,blocked);$finish;
  end
  initial begin #5000000;$fatal(1,"pool watchdog");end
endmodule
