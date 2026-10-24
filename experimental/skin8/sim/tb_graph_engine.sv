`timescale 1ns/1ps
module tb_graph_engine;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0,start=0;
  logic [31:0] cfg_source=32'h10000,cfg_weights=32'h20000,cfg_destination=32'h40000;
  logic [31:0] cfg_source2=32'h18000;
  logic [1:0] cfg_mode=0;
  logic cfg_concat=0,cfg_table2=1;
  logic [4:0] cfg_branch_groups;
  logic [8:0] cfg_width=8,cfg_height=8,cfg_y=0,cfg_rows=8;
  logic [4:0] cfg_groups;
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
  logic [63:0] memory [0:2097151],golden [0:2097151];
  integer pending_r=0,rindex=0,pending_w=0,seen=0,seed=2026,cycles=0,checked=0,case_id=0;
  integer fd,ncases,groups,kernel,relui,rc,modei,concati,bgroups,reusei,epoch,load_luts,li;
  integer wi,hi,yi,rowsi,model_start,layer_start,last_layer,feature_reads=0;
  logic [31:0] read_base,write_base;
  logic consumed_r=0,consumed_b=0,need_response=0;
  string folder;
  always @(posedge clk) begin
    cycles=cycles+1;consumed_r=rvalid && rready;consumed_b=bvalid && bready;
    if(resetn) begin
      if(arvalid && arready) begin
        if(pending_r || arsize!=3 || arburst!=1 || arlen>15 || araddr[11:0]+(arlen+1)*8>4096)
          $fatal(1,"bad engine read burst");
        pending_r=arlen+1;rindex=0;read_base=araddr;
        if(araddr+(arlen+1)*8>32'h1000000) $fatal(1,"read outside DDR model");

      end
      if(consumed_r) begin pending_r=pending_r-1;rindex=rindex+1;end
      if(awvalid && awready) begin
        if(pending_w || need_response || awsize!=3 || awburst!=1 || awlen>15 || awaddr!=cfg_destination+seen*8)
          $fatal(1,"bad engine write burst");
        pending_w=awlen+1;write_base=awaddr;
      end
      if(wvalid && wready) begin
        if(pending_w==0 || seen>=cfg_width*cfg_rows || wdata!==golden[(cfg_destination>>3)+seen] || wstrb!==8'hff || wlast!==(pending_w==1))
          $fatal(1,"DDR golden mismatch case=%0d word=%0d expected=%h got=%h",case_id,seen,golden[(cfg_destination>>3)+seen],wdata);
        memory[(cfg_destination>>3)+seen]=wdata;
        seen=seen+1;pending_w=pending_w-1;checked=checked+1;
        if(pending_w==0) need_response=1;
      end
      if(consumed_b) need_response=0;
    end
    if(cycles>150000000) $fatal(1,"engine timeout state=%0d",dut.state);
  end
  always @(negedge clk) begin
    arready=$unsigned($random(seed))%3!=0;awready=$unsigned($random(seed))%3!=0;wready=$unsigned($random(seed))%3!=0;
    if(!rvalid || consumed_r) begin
      rvalid=pending_r>0 && $unsigned($random(seed))%4!=0;
      rdata=memory[(read_base>>3)+rindex];
      if(rvalid && $isunknown(rdata)) $fatal(1,"read uninitialized DDR at %h",read_base+rindex*8);
      rlast=pending_r==1;rresp=0;
    end
    if(!bvalid || consumed_b) begin
      bvalid=need_response && $unsigned($random(seed))%4!=0;bresp=0;
    end
  end
  task automatic launch(input integer expect_error);
    @(negedge clk);seen=0;start=1;cfg_page=cfg_feature_id%2;
    @(negedge clk);start=0;
    while(!done) @(negedge clk);
    if(error!==expect_error[0]) $fatal(1,"wrong engine completion status");
    if(!expect_error && seen!=cfg_width*cfg_rows) $fatal(1,"missing DDR output");
    if(pending_r || pending_w || need_response) $fatal(1,"completion with outstanding bus work");
    repeat(3) @(negedge clk);
  endtask
  initial begin
    repeat(4) @(negedge clk);resetn=1;
    for(int model=0;model<2;model++) begin
      folder=model==0 ? "../model_vectors/graph_skin/" : "../model_vectors/graph_camus/";
      for(int i=0;i<2097152;i++) begin memory[i]='x;golden[i]='x;end
      $readmemh({folder,"initial.mem"},memory);
      $readmemh({folder,"golden.mem"},golden);
      fd=$fopen({folder,"commands.txt"},"r");if(!fd) $fatal(1,"missing graph commands");
      rc=$fscanf(fd,"%d",ncases);if(rc!=1) $fatal(1,"missing command count");
      model_start=cycles;layer_start=cycles;last_layer=0;
      for(case_id=0;case_id<ncases;case_id++) begin
        rc=$fscanf(fd,"%d %d %d %d %d %d %d %d %d %d %d %d %d %d %h %h %h %h %h %h",
          li,groups,kernel,relui,wi,hi,yi,rowsi,modei,concati,bgroups,reusei,epoch,load_luts,
          cfg_source,cfg_source2,cfg_weights,cfg_destination,cfg_bias,cfg_multiplier);
        if(rc!=20) $fatal(1,"bad graph descriptor %0d",case_id);
        if(li!=last_layer) begin
          $display("GRAPH model=%0d layer=%0d cycles=%0d",model,last_layer,cycles-layer_start);
          last_layer=li;layer_start=cycles;
        end
        cfg_groups=groups;cfg_k3=kernel==3;cfg_relu=relui;cfg_mode=modei;cfg_concat=concati;
        cfg_branch_groups=bgroups;cfg_reuse_features=reusei;cfg_feature_id=epoch+model*10000;
        cfg_width=wi;cfg_height=hi;cfg_y=yi;cfg_rows=rowsi;
        if(load_luts) begin
          for(int t=0;t<2;t++) for(int v=0;v<256;v++) begin
            @(negedge clk);lut_valid=1;lut_table=t;lut_index=v;rc=$fscanf(fd,"%h",lut_value);
            if(rc!=1) $fatal(1,"bad graph LUT");
          end
          @(negedge clk);lut_valid=0;
        end
        launch(0);
      end
      $display("GRAPH model=%0d layer=%0d cycles=%0d",model,last_layer,cycles-layer_start);
      $display("GRAPH model=%0d total_cycles=%0d descriptors=%0d",model,cycles-model_start,ncases);
      $fclose(fd);
    end
    $display("PASS graph_engine checked=%0d models=2 full_input_256=1 intermediate_injection=0",checked);$finish;
  end
endmodule
