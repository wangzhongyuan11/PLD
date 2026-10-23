`timescale 1ns/1ps
module tb_input_loader #(parameter integer P=4);
  localparam integer WORDS=(P==4?4608:3584),AW=$clog2(WORDS);
  logic aclk=0,aresetn=0,start=0;
  logic [31:0] cfg_source0=0,cfg_source1=0;
  logic [15:0] cfg_image_h=0,cfg_image_w=0,cfg_y=0,cfg_x=0,cfg_oh=0,cfg_ow=0;
  logic [7:0] cfg_ci=0,cfg_split=0;
  logic cfg_kernel3=0,cfg_cat=0,cfg_slot=0,lut_we;
  logic [8:0] lut_addr;
  logic [7:0] lut_data;
  logic busy,done;
  logic [7:0] error;
  logic [71:0] cmd_data;
  logic cmd_valid,cmd_ready,status_valid,status_ready;
  logic [7:0] status_data,dma_keep;
  logic [63:0] dma_data;
  logic dma_last,dma_valid,dma_ready;
  logic load_slot,load_valid,load_ready=0;
  logic [$clog2(P)-1:0] load_bank;
  logic [AW-1:0] load_addr;
  logic [63:0] load_data;
  logic [7:0] load_keep;
  logic [63:0] expected[int unsigned];
  bit seen[int unsigned];
  logic [7:0] lut_values[512];
  int fd,ef,rc,ncases,nexpected,case_index=0,writes=0,supplied_words,previous_words,expected_reads;
  int unsigned key,rng=32'hacdc510a;
  logic [63:0] value;
  logic [64+AW+$clog2(P):0] held;
  bit stalled=0;
  logic lut_start=0,lut_done,lut_busy;
  logic [7:0] lut_error;
  integer lut_writes=0;
  logic [71:0] input_cmd,lut_cmd;
  logic input_cmd_valid,input_cmd_ready,input_status_valid,input_status_ready;
  logic lut_cmd_valid,lut_cmd_ready,lut_status_valid,lut_status_ready;
  logic [7:0] input_status,lut_status,input_keep,lut_keep;
  logic [63:0] input_data,lut_payload;
  logic input_last,input_valid,input_ready,lut_last,lut_valid,lut_ready;
  logic [1:0] c_cmd_ready,c_last,c_valid,c_status_valid;
  logic [127:0] c_data;
  logic [15:0] c_keep,c_status;
  logic arbiter_busy,arbiter_error;
  input_loader #(.P(P)) dut(.cmd_data(input_cmd),.cmd_valid(input_cmd_valid),.cmd_ready(input_cmd_ready),
    .status_data(input_status),.status_valid(input_status_valid),.status_ready(input_status_ready),
    .dma_data(input_data),.dma_keep(input_keep),.dma_last(input_last),.dma_valid(input_valid),.dma_ready(input_ready),.*);
  lut_loader table_loader(.aclk,.aresetn,.start(lut_start),.cfg_address(32'h01000000),
    .busy(lut_busy),.done(lut_done),.error(lut_error),.cmd_data(lut_cmd),.cmd_valid(lut_cmd_valid),.cmd_ready(lut_cmd_ready),
    .status_data(lut_status),.status_valid(lut_status_valid),.status_ready(lut_status_ready),
    .dma_data(lut_payload),.dma_keep(lut_keep),.dma_last(lut_last),.dma_valid(lut_valid),.dma_ready(lut_ready),
    .lut_we,.lut_addr,.lut_data);
  dma_read_arbiter #(.N(2)) arbiter(.aclk,.aresetn,.c_cmd_data({input_cmd,lut_cmd}),
    .c_cmd_valid({input_cmd_valid,lut_cmd_valid}),.c_cmd_ready,.c_data,.c_keep,.c_last,.c_valid,
    .c_ready({input_ready,lut_ready}),.c_status,.c_status_valid,.c_status_ready({input_status_ready,lut_status_ready}),
    .cmd_data,.cmd_valid,.cmd_ready,.data(dma_data),.keep(dma_keep),.last(dma_last),.valid(dma_valid),.ready(dma_ready),
    .status(status_data),.status_valid,.status_ready,.busy(arbiter_busy),.protocol_error(arbiter_error));
  assign {input_cmd_ready,lut_cmd_ready}=c_cmd_ready;
  assign {input_data,lut_payload}=c_data;assign {input_keep,lut_keep}=c_keep;
  assign {input_last,lut_last}=c_last;assign {input_valid,lut_valid}=c_valid;
  assign {input_status,lut_status}=c_status;assign {input_status_valid,lut_status_valid}=c_status_valid;
  datamover_source source(.aclk,.aresetn,.cmd_data,.cmd_valid,.cmd_ready,
    .data(dma_data),.keep(dma_keep),.last(dma_last),.valid(dma_valid),.ready(dma_ready),
    .status_data,.status_valid,.status_ready,.supplied_words);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk)begin rng=next_rng(rng);load_ready=aresetn&&(rng[0]||rng[4]);end
  always @(posedge aclk)if(aresetn)begin
    if(arbiter_error)$fatal(1,"input/LUT arbiter error");
    if(lut_we)begin
      if(lut_data!==lut_values[lut_addr]||lut_addr!=lut_writes)$fatal(1,"DMA LUT order/value mismatch");
      lut_writes++;
    end
    if(stalled&&(!load_valid||{load_slot,load_bank,load_addr,load_data}!==held))$fatal(1,"input loader bank write unstable");
    stalled=load_valid&&!load_ready;held={load_slot,load_bank,load_addr,load_data};
    if(load_valid&&load_ready)begin
      key=(case_index<<20)|(int'(load_bank)<<16)|load_addr;
      if(!expected.exists(key))$fatal(1,"unexpected input bank address case=%0d bank=%0d addr=%0d",case_index,load_bank,load_addr);
      if(seen.exists(key)||load_keep!=8'hff||load_slot!=cfg_slot)$fatal(1,"duplicate/malformed bank write");
      if(load_data!==expected[key])$fatal(1,"input bank mismatch P=%0d case=%0d bank=%0d addr=%0d actual=%h expected=%h",P,case_index,load_bank,load_addr,load_data,expected[key]);
      seen[key]=1;writes++;
    end
  end else stalled=0;
  initial begin
    fd=$fopen("input_cases.txt","r");ef=$fopen("banks_expected.txt","r");
    if(!fd||!ef)$fatal(1,"missing input-loader vectors");
    rc=$fscanf(fd,"%d\n",ncases);rc=$fscanf(ef,"%d\n",nexpected);
    for(int j=0;j<nexpected;j++)begin
      rc=$fscanf(ef,"%h %h\n",key,value);if(rc!=2||expected.exists(key))$fatal(1,"bad bank oracle");expected[key]=value;
    end
    $readmemh("lut.txt",lut_values);
    repeat(4)@(negedge aclk);aresetn=1;
    @(negedge aclk);lut_start=1;@(posedge aclk);@(negedge aclk);lut_start=0;
    wait(lut_done);@(negedge aclk);
    if(lut_error||lut_writes!=512)$fatal(1,"DMA LUT load failed");
    for(case_index=0;case_index<ncases;case_index++)begin
      @(negedge aclk);
      rc=$fscanf(fd,"%h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",cfg_source0,cfg_source1,cfg_image_h,cfg_image_w,cfg_y,cfg_x,cfg_oh,cfg_ow,cfg_ci,cfg_split,cfg_kernel3,cfg_cat,cfg_slot,expected_reads);
      if(rc!=14)$fatal(1,"bad input config");previous_words=supplied_words;start=1;
      @(posedge aclk);@(negedge aclk);start=0;wait(done);@(negedge aclk);
      if(error)$fatal(1,"input loader error=%h case=%0d",error,case_index);
      if(supplied_words-previous_words!=expected_reads)$fatal(1,"CAT row cache read count %0d/%0d",supplied_words-previous_words,expected_reads);
    end
    if(writes!=nexpected)$fatal(1,"missing bank writes %0d/%0d",writes,nexpected);
    cfg_ci=0;start=1;@(posedge aclk);@(negedge aclk);start=0;
    if(!done||error!=1)$fatal(1,"invalid input config not rejected");
    aresetn=0;repeat(3)@(negedge aclk);aresetn=1;@(negedge aclk);
    if(error||busy||load_valid)$fatal(1,"input loader reset failed");
    $display("TB_PASS input_loader P=%0d cases=%0d bank_writes=%0d DMA_LUT=512 normal_CAT_halo_nearest_cache=PASS",P,ncases,writes);$finish;
  end
  initial begin #20000000;$fatal(1,"input loader watchdog");end
endmodule
module tb_input_loader_p4;
  tb_input_loader #(.P(4)) test();
endmodule
module tb_input_loader_p6;
  tb_input_loader #(.P(6)) test();
endmodule
