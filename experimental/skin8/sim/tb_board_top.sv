`timescale 1ns/1ps
module tb_board_top;
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
  logic [7:0] s_awaddr=0,s_araddr=0;
  logic s_awvalid=0,s_awready,s_wvalid=0,s_wready,s_bvalid,s_bready=0;
  logic [31:0] s_wdata=0,s_rdata;
  logic [3:0] s_wstrb=15;
  logic [1:0] s_bresp,s_rresp;
  logic s_arvalid=0,s_arready,s_rvalid,s_rready=0;
  logic [1023:0] command;
  logic [7:0] tables [0:511];
  logic need_lut=1;
  board_top dut(.clk,.resetn,.s_axi_awprot(3'b0),.s_axi_arprot(3'b0),.m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arvalid(arvalid),.m_axi_arready(arready),.m_axi_rdata(rdata),.m_axi_rresp(rresp),.m_axi_rlast(rlast),.m_axi_rvalid(rvalid),.m_axi_rready(rready),.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),.m_axi_awsize(awsize),.m_axi_awburst(awburst),.m_axi_awvalid(awvalid),.m_axi_awready(awready),.m_axi_wdata(wdata),.m_axi_wstrb(wstrb),.m_axi_wlast(wlast),.m_axi_wvalid(wvalid),.m_axi_wready(wready),.m_axi_bresp(bresp),.m_axi_bvalid(bvalid),.m_axi_bready(bready),.s_axi_awaddr(s_awaddr),.s_axi_awvalid(s_awvalid),.s_axi_awready(s_awready),.s_axi_wdata(s_wdata),.s_axi_wstrb(s_wstrb),.s_axi_wvalid(s_wvalid),.s_axi_wready(s_wready),.s_axi_bresp(s_bresp),.s_axi_bvalid(s_bvalid),.s_axi_bready(s_bready),.s_axi_araddr(s_araddr),.s_axi_arvalid(s_arvalid),.s_axi_arready(s_arready),.s_axi_rdata(s_rdata),.s_axi_rresp(s_rresp),.s_axi_rvalid(s_rvalid),.s_axi_rready(s_rready));
  logic [63:0] inputs [0:8191],inputs2 [0:2047],weights [0:2047],golden [0:63];
  integer pending_r=0,rindex=0,pending_w=0,seen=0,seed=2026,cycles=0,checked=0,case_id=0;
  integer fd,ncases,groups,kernel,relui,rc,modei,concati,bgroups,nfirst,nsecond;
  integer feature_reads=0,reads_before;
  logic [31:0] read_base;
  logic consumed_r=0,consumed_b=0,need_response=0,inject_read_error=0,inject_write_error=0,inject_second_error=0;
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
    if(cycles>3000000) $fatal(1,"engine timeout state=%0d",dut.engine.controller.state);
  end
  always @(negedge clk) begin
    arready=$unsigned($random(seed))%3!=0;awready=$unsigned($random(seed))%3!=0;wready=$unsigned($random(seed))%3!=0;
    if(!rvalid || consumed_r) begin
      rvalid=pending_r>0 && $unsigned($random(seed))%4!=0;
      if(read_base>=32'h60000) rdata=command[((read_base-32'h60000)/8+rindex)*64+:64];
      else if(read_base>=32'h50000) begin
        for(int lane=0;lane<8;lane++) rdata[lane*8+:8]=tables[read_base-32'h50000+rindex*8+lane];
      end
      else if(read_base>=cfg_weights) rdata=weights[(read_base-cfg_weights)/8+rindex];
      else if(read_base>=cfg_source2) rdata=inputs2[(read_base-cfg_source2)/8+rindex];
      else rdata=inputs[(read_base-cfg_source)/8+rindex];
      rlast=pending_r==1;rresp=(inject_read_error || (inject_second_error && read_base>=cfg_source2 && read_base<cfg_weights))?2:0;
    end
    if(!bvalid || consumed_b) begin
      bvalid=need_response && $unsigned($random(seed))%4!=0;bresp=inject_write_error?2:0;
    end
  end
  task automatic write_reg(input logic [7:0] addr,input logic [31:0] value);
    @(negedge clk);s_awaddr=addr;s_awvalid=1;
    do @(posedge clk); while(!s_awready);
    @(negedge clk);s_awvalid=0;s_wdata=value;s_wvalid=1;
    do @(posedge clk); while(!s_wready);
    @(negedge clk);s_wvalid=0;s_bready=1;
    do @(posedge clk); while(!s_bvalid);
    if(s_bresp!=0) $fatal(1,"MMIO write error");
    @(negedge clk);s_bready=0;
  endtask
  task automatic read_status(output logic [31:0] value);
    @(negedge clk);s_araddr=0;s_arvalid=1;
    do @(posedge clk); while(!s_arready);
    @(negedge clk);s_arvalid=0;s_rready=1;
    do @(posedge clk); while(!s_rvalid);
    value=s_rdata;if(s_rresp!=0) $fatal(1,"MMIO read error");
    @(negedge clk);s_rready=0;
  endtask
  task automatic launch(input integer expect_error);
    logic [31:0] status;
    @(negedge clk);seen=0;command=0;
    command[0+:32]=cfg_source;command[32+:32]=cfg_source2;command[64+:32]=cfg_weights;command[96+:32]=cfg_destination;
    command[128+:16]=cfg_width;command[144+:16]=cfg_height;command[160+:16]=cfg_y;command[176+:16]=cfg_rows;
    command[192+:8]=cfg_groups;command[200+:8]=cfg_branch_groups;command[208+:8]=cfg_mode;
    command[216+:8]={need_lut,cfg_reuse_features,cfg_concat,case_id[0],cfg_table2,cfg_table,cfg_relu,cfg_k3};
    command[224+:32]=cfg_feature_id;command[256+:32]=32'h50000;
    command[384+:256]=cfg_bias;command[640+:256]=cfg_multiplier;
    write_reg(8'h10,32'h60000);write_reg(8'h14,1);write_reg(0,1);
    do read_status(status); while(!status[2] || status[1]);
    need_lut=0;
    if(status[3]!==expect_error[0]) $fatal(1,"wrong MMIO completion status %h",status);
    if(!expect_error && seen!=64) $fatal(1,"missing DDR output");
    if(pending_r || pending_w || need_response) $fatal(1,"completion with outstanding bus work");
    repeat(3) @(negedge clk);
  endtask
  initial begin
    fd=$fopen("../model_vectors/transformed_cases.txt","r");
    if(!fd) $fatal(1,"missing engine vectors");
    rc=$fscanf(fd,"%d",ncases);if(rc!=1) $fatal(1,"bad case count");
    repeat(4) @(negedge clk);resetn=1;
    for(case_id=0;case_id<ncases;case_id++) begin
      rc=$fscanf(fd,"%d %d %h %h %d %d %d %d %d %d",groups,kernel,cfg_bias,cfg_multiplier,relui,modei,concati,bgroups,nfirst,nsecond);
      if(rc!=10) $fatal(1,"bad engine header");
      cfg_groups=groups;cfg_k3=kernel==3;cfg_relu=relui;cfg_feature_id=case_id;cfg_mode=modei;cfg_concat=concati;cfg_branch_groups=bgroups;cfg_reuse_features=0;need_lut=1;
      for(int t=0;t<2;t++) for(int v=0;v<256;v++) begin
        @(negedge clk);lut_valid=1;lut_table=t;lut_index=v;rc=$fscanf(fd,"%h",lut_value);
        if(rc!=1) $fatal(1,"LUT vector");tables[t*256+v]=lut_value;
      end
      @(negedge clk);lut_valid=0;
      for(int n=0;n<nfirst;n++) begin rc=$fscanf(fd,"%h",inputs[n]);if(rc!=1) $fatal(1,"input vector");end
      for(int n=0;n<nsecond;n++) begin rc=$fscanf(fd,"%h",inputs2[n]);if(rc!=1) $fatal(1,"second input vector");end
      for(int n=0;n<8*kernel*kernel*groups;n++) begin rc=$fscanf(fd,"%h",weights[n]);if(rc!=1) $fatal(1,"weight vector");end
      for(int n=0;n<64;n++) begin rc=$fscanf(fd,"%h",golden[n]);if(rc!=1) $fatal(1,"golden vector");end
      launch(0);
      reads_before=feature_reads;cfg_reuse_features=1;launch(0);
      if(feature_reads!=reads_before) $fatal(1,"transformed feature reuse reread");
      if(cfg_concat) begin
        cfg_source2=cfg_source2+8;launch(1);cfg_source2=cfg_source2-8;
        cfg_reuse_features=0;inject_second_error=1;launch(1);inject_second_error=0;
        cfg_reuse_features=1;launch(1); // Partial concat must never become reusable.
        cfg_reuse_features=0;launch(0);
      end
      cfg_reuse_features=0;
    end
    $fclose(fd);
    $display("PASS board_top checked=%0d blocks=%0d frozen_layers=30 transformed_branches=1 guarded_reuse=1",checked,ncases);$finish;
  end
endmodule
