`timescale 1ns/1ps
module tb_tile_job_queue;
  logic aclk=0,aresetn=0,clear_cache=0,s_valid=0,s_ready;
  logic [511:0] s_descriptor=0,load_descriptor,run_descriptor;
  logic load_input_slot,load_weight_slot,run_input_slot,run_weight_slot;
  logic input_start,parameter_start,lut_start,run_start;
  logic input_done=0,parameter_done=0,lut_done=0,core_done=0,writer_done=0;
  logic [7:0] input_error=0,parameter_error=0,lut_error=0,writer_error=0;
  logic [2:0] core_error=0;
  logic [31:0] completion_tag,error_tag,input_hits,parameter_hits,overlap_cycles;
  logic completion_valid,completion_ready=0,busy,fault;logic [15:0] error;
  int it=0,pt=0,lt=0,ct=0,wt=0,submitted=0,completed=0,cycles=0;
  int input_loads=0,param_loads=0,table_loads=0,runs=0;
  bit active_input_slot,active_weight_slot,compute_active=0;
  int unsigned rng=32'h1cdcfade;
  logic [511:0] descriptors[10];
  tile_job_queue dut(.*);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk)begin
    rng=next_rng(rng);completion_ready=aresetn&&(rng[1]||rng[5]);
    if(aresetn)begin
      input_done=it==1;parameter_done=pt==1;lut_done=lt==1;core_done=ct==1;writer_done=wt==1;
      if(it>0)it--;if(pt>0)pt--;if(lt>0)lt--;if(ct>0)ct--;if(wt>0)wt--;
    end else begin input_done=0;parameter_done=0;lut_done=0;core_done=0;writer_done=0;end
  end
  always @(posedge aclk)if(aresetn)begin
    cycles++;
    if(input_start)begin
      if(it||lt||(compute_active&&load_input_slot==active_input_slot))$fatal(1,"input cache ownership/start violation");
      if(load_descriptor[380]&&completed!=load_descriptor[415:384])$fatal(1,"barrier prefetch before old completion");
      input_loads++;it=17;
    end
    if(parameter_start)begin
      if(pt||(compute_active&&load_weight_slot==active_weight_slot))$fatal(1,"weight cache ownership/start violation");
      if(load_descriptor[380]&&completed!=load_descriptor[415:384])$fatal(1,"barrier parameter prefetch");
      param_loads++;pt=23;
    end
    if(lut_start)begin
      if(lt||it)$fatal(1,"LUT changed during an input operation");
      table_loads++;lt=29;
    end
    if(run_start)begin
      if(compute_active||run_descriptor[415:384]!=runs)$fatal(1,"unordered/overlapping compute launch");
      compute_active=1;active_input_slot=run_input_slot;active_weight_slot=run_weight_slot;
      ct=35;wt=75;runs++;
    end
    if(completion_valid&&completion_ready)begin
      if(completion_tag!=completed||ct||wt)$fatal(1,"early/unordered job completion");
      compute_active=0;completed++;
    end
    if(cycles>6000)$fatal(1,"queue deadlock");
  end
  task automatic send(input logic [511:0] d);
    @(negedge aclk);s_descriptor=d;s_valid=1;
    do @(posedge aclk);while(!s_ready);
    @(negedge aclk);s_valid=0;submitted++;
  endtask
  task automatic reset;
    @(negedge aclk);aresetn=0;s_valid=0;clear_cache=0;
    repeat(4)@(negedge aclk);
    it=0;pt=0;lt=0;ct=0;wt=0;compute_active=0;aresetn=1;
  endtask
  initial begin
    for(int j=0;j<10;j++)begin
      descriptors[j]=0;
      descriptors[j][31:0]=32'h10000000;descriptors[j][95:64]=32'h11000000;
      descriptors[j][159:128]=32'h12000000;descriptors[j][191:160]=64;
      descriptors[j][287:256]=32'h00080008;descriptors[j][351:320]=32'h00080008;
      descriptors[j][383:352]=32'h03000810;descriptors[j][415:384]=j;
      descriptors[j][447:416]=1;descriptors[j][511:480]=1;
    end
    descriptors[1][95:64]+=4096;
    descriptors[3][303:288]=2;descriptors[3][335:320]=4;
    descriptors[4][380]=1;
    for(int j=5;j<8;j++)begin
      descriptors[j][63:32]=32'h13000000;descriptors[j][127:96]=32'h14000000;
      descriptors[j][383:352]=32'h07100830;
    end
    descriptors[6][95:64]+=4096;
    descriptors[7][303:288]=2;descriptors[7][335:320]=4;
    descriptors[8][447:416]=2;
    descriptors[9][511:480]=2;
    reset();
    for(int j=0;j<10;j++)send(descriptors[j]);
    wait(completed==10);repeat(5)@(negedge aclk);
    if(fault||busy||runs!=10||input_loads!=6||param_loads!=6||table_loads!=1||
       input_hits!=4||parameter_hits!=4||overlap_cycles==0)
      $fatal(1,"queue accounting inputs=%0d params=%0d tables=%0d hits=%0d/%0d overlap=%0d",input_loads,param_loads,table_loads,input_hits,parameter_hits,overlap_cycles);
    // Cache clear is legal only once all checked completion records are consumed.
    clear_cache=1;@(negedge aclk);clear_cache=0;
    descriptors[0][415:384]=10;send(descriptors[0]);wait(completed==11);
    if(input_loads!=7||param_loads!=7)$fatal(1,"idle clear did not invalidate both caches");
    reset();descriptors[0][383:376]=8'h83;descriptors[0][415:384]=123;
    send(descriptors[0]);repeat(4)@(negedge aclk);
    if(!fault||error!=16'h0101||error_tag!=123||s_ready||completion_valid)$fatal(1,"invalid descriptor not terminal");
    reset();descriptors[0][383:376]=3;descriptors[0][415:384]=11;
    send(descriptors[0]);wait(input_start);@(negedge aclk);input_error=8'h08;
    repeat(3)@(negedge aclk);input_error=0;
    if(!fault||error!=16'h0208||error_tag!=11||s_ready)$fatal(1,"loader fault not preserved");
    repeat(10)@(negedge aclk);
    $display("TB_PASS tile_job_queue ordered=11 faults=2 ownership=PASS barriers=PASS hits=PASS overlap=PASS");$finish;
  end
endmodule
