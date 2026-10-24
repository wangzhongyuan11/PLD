`timescale 1ns/1ps
module tb_stripe_engine;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0,start=0;
  logic [31:0] cfg_source=32'h10000,cfg_weights=32'h20000,cfg_destination=32'h40000;
  logic [31:0] cfg_source2=32'h18000;
  logic [1:0] cfg_mode=0;
  logic cfg_concat=0,cfg_table2=1;
  wire [4:0] cfg_branch_groups;
  logic [8:0] cfg_width=8,cfg_height=8,cfg_y=0,cfg_rows=8;
  logic [4:0] cfg_groups;
  assign cfg_branch_groups=cfg_groups;
  logic cfg_k3,cfg_relu,cfg_table=0,cfg_page=0;
  logic cfg_reuse_features=0;
  logic [31:0] cfg_feature_id=0;
  logic [255:0] cfg_bias,cfg_multiplier;
  logic busy,done,error,lut_table=0,lut_valid=0,lut_ready;
  logic [7:0] lut_index=0,lut_value=0;
  logic [31:0] araddr,awaddr;
  logic [7:0] arlen,awlen,wstrb;
  logic [2:0] arsize,awsize;
  logic [1:0] arburst,awburst,rresp=0,bresp=0;
  logic arvalid,arready=0,rlast=0,rvalid=0,rready;
  logic awvalid,awready=0,wlast,wvalid,wready=0,bvalid=0,bready;
  logic [63:0] rdata=0,wdata;
  stripe_engine dut(.*);
  logic [63:0] inputs [0:2047],weights [0:2047],golden [0:63];
  integer pending_r=0,rindex=0,pending_w=0,seen=0,seed=2026,cycles=0,checked=0,case_id=0;
  integer fd,ncases,groups,kernel,relui,rc;
  integer feature_reads=0,reads_before;
  logic [31:0] read_base;
  logic consumed_r=0,consumed_b=0,need_response=0,inject_read_error=0,inject_write_error=0;
  always @(posedge clk) begin
    cycles=cycles+1;consumed_r=rvalid && rready;consumed_b=bvalid && bready;
    if(resetn) begin
      if(arvalid && arready) begin
        if(pending_r || arsize!=3 || arburst!=1 || arlen>15 || araddr[11:0]+(arlen+1)*8>4096)
          $fatal(1,"bad engine read burst");
        pending_r=arlen+1;rindex=0;read_base=araddr;
        if(araddr<cfg_weights) feature_reads=feature_reads+arlen+1;
      end
      if(consumed_r) begin pending_r=pending_r-1;rindex=rindex+1;end
      if(awvalid && awready) begin
        if(pending_w || need_response || awsize!=3 || awburst!=1 || awlen>15 || awaddr!=cfg_destination+seen*8)
          $fatal(1,"bad engine write burst");
        pending_w=awlen+1;
      end
      if(wvalid && wready) begin
        if(pending_w==0 || seen>=64 || wdata!==golden[seen] || wstrb!==8'hff || wlast!==(pending_w==1))
          $fatal(1,"DDR golden mismatch case=%0d word=%0d expected=%h got=%h",case_id,seen,golden[seen],wdata);
        seen=seen+1;pending_w=pending_w-1;checked=checked+1;
        if(pending_w==0) need_response=1;
      end
      if(consumed_b) need_response=0;
    end
    if(cycles>3000000) $fatal(1,"engine timeout state=%0d",dut.state);
  end
  always @(negedge clk) begin
    arready=$unsigned($random(seed))%3!=0;awready=$unsigned($random(seed))%3!=0;wready=$unsigned($random(seed))%3!=0;
    if(!rvalid || consumed_r) begin
      rvalid=pending_r>0 && $unsigned($random(seed))%4!=0;
      if(read_base>=cfg_weights) rdata=weights[(read_base-cfg_weights)/8+rindex];
      else rdata=inputs[(read_base-cfg_source)/8+rindex];
      rlast=pending_r==1;rresp=inject_read_error?2:0;
    end
    if(!bvalid || consumed_b) begin
      bvalid=need_response && $unsigned($random(seed))%4!=0;bresp=inject_write_error?2:0;
    end
  end
  task automatic launch(input integer expect_error);
    @(negedge clk);seen=0;start=1;cfg_page=case_id%2;
    @(negedge clk);start=0;
    while(!done) @(negedge clk);
    if(error!==expect_error[0]) $fatal(1,"wrong engine completion status");
    if(!expect_error && seen!=64) $fatal(1,"missing DDR output");
    if(pending_r || pending_w || need_response) $fatal(1,"completion with outstanding bus work");
    repeat(3) @(negedge clk);
  endtask
  initial begin
    fd=$fopen("../model_vectors/engine_cases.txt","r");
    if(!fd) $fatal(1,"missing engine vectors");
    rc=$fscanf(fd,"%d",ncases);if(rc!=1) $fatal(1,"bad case count");
    repeat(4) @(negedge clk);resetn=1;
    for(int v=0;v<256;v++) begin @(negedge clk);lut_valid=1;lut_index=v;lut_value=v;end
    @(negedge clk);lut_valid=0;
    for(case_id=0;case_id<ncases;case_id++) begin
      rc=$fscanf(fd,"%d %d %h %h %d",groups,kernel,cfg_bias,cfg_multiplier,relui);
      if(rc!=5) $fatal(1,"bad engine header");
      cfg_groups=groups;cfg_k3=kernel==3;cfg_relu=relui;cfg_feature_id=case_id;
      for(int n=0;n<groups*64;n++) begin rc=$fscanf(fd,"%h",inputs[n]);if(rc!=1) $fatal(1,"input vector");end
      for(int n=0;n<8*kernel*kernel*groups;n++) begin rc=$fscanf(fd,"%h",weights[n]);if(rc!=1) $fatal(1,"weight vector");end
      for(int n=0;n<64;n++) begin rc=$fscanf(fd,"%h",golden[n]);if(rc!=1) $fatal(1,"golden vector");end
      launch(0);
    end
    inject_read_error=1;launch(1);inject_read_error=0;launch(0);
    inject_write_error=1;launch(1);inject_write_error=0;launch(0);
    reads_before=feature_reads;cfg_reuse_features=1;launch(0);
    if(feature_reads!=reads_before) $fatal(1,"feature reuse reread DDR inputs");
    cfg_feature_id=cfg_feature_id+1;launch(1);cfg_feature_id=cfg_feature_id-1;
    @(negedge clk);lut_valid=1;lut_index=5;lut_value=5;
    @(negedge clk);lut_valid=0;
    launch(1); // Even an identical LUT write invalidates resident mapped bytes.
    cfg_reuse_features=0;launch(0);
    $fclose(fd);
    $display("PASS stripe_engine checked=%0d blocks=%0d frozen_layers=30 error_recovery=1 guarded_reuse=1",checked,ncases);$finish;
  end
endmodule
