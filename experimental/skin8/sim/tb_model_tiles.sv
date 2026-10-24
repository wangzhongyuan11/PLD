`timescale 1ns/1ps
module tb_model_tiles;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0,start=0,cfg_page=0,cfg_k3=0;
  logic [8:0] cfg_iw=0,cfg_ih=0;
  logic [4:0] cfg_ci_groups=0;
  logic [255:0] cfg_bias=0;
  logic busy,error,done;
  logic f_page=0,f_bank=0,f_valid=0,f_ready;
  logic [11:0] f_addr=0;
  logic [63:0] f_data=0,w_data=0;
  logic [2:0] w_lane=0;
  logic [7:0] w_addr=0;
  logic w_valid=0,w_ready;
  logic [511:0] m_acc;
  logic [15:0] m_tag;
  logic m_last,m_second_valid,m_overflow,m_valid,m_ready;
  logic [255:0] multipliers;
  logic relu;
  logic [127:0] q_data;
  logic [15:0] q_tag;
  logic q_last,q_second,q_overflow,q_valid,q_ready=0;
  tile_int8 dut(.clk,.resetn,.start,.cfg_page,.cfg_k3,.cfg_relu(relu),.cfg_iw,.cfg_ih,.cfg_ci_groups,
    .cfg_bias,.cfg_multiplier(multipliers),.busy,.error,.done,
    .f_page,.f_bank,.f_addr,.f_data,.f_valid,.f_ready,.w_lane,.w_addr,.w_data,.w_valid,.w_ready,
    .m_data(q_data),.m_tag(q_tag),.m_last(q_last),.m_second_valid(q_second),.m_overflow(q_overflow),.m_valid(q_valid),.m_ready(q_ready));
  // Observe the internal arithmetic boundary as well as the public INT8 output.
  assign m_acc=dut.acc;assign m_tag=dut.tag;assign m_valid=dut.valid;assign m_ready=dut.ready;
  assign m_last=dut.last;assign m_second_valid=dut.second_valid;assign m_overflow=dut.overflow;
  logic [511:0] expected [0:31];
  logic [127:0] expected_quant [0:31];
  integer fd,ncases,case_id,iw,ih,groups,kernel,bank,addr,lane,rc,seen=0,q_seen=0,checked=0,seed=12345,cycles=0,relui;
  logic held=0;
  logic [146:0] held_data;
  always @(posedge clk) begin
    cycles=cycles+1;
    if(resetn) begin
      if(held && (!q_valid || {q_data,q_tag,q_last,q_second,q_overflow}!==held_data)) $fatal(1,"quant group changed under stall");
      held=q_valid && !q_ready;held_data={q_data,q_tag,q_last,q_second,q_overflow};
      if(q_valid && q_ready) begin
        if(q_tag!=q_seen || q_data!==expected_quant[q_seen] || q_last!==(q_seen==31) || q_second!==1 || q_overflow!==0)
          $fatal(1,"quantized frozen fixture mismatch case=%0d pair=%0d expected=%h got=%h",case_id,q_seen,expected_quant[q_seen],q_data);
        q_seen=q_seen+1;
      end
    end
    if(resetn && m_valid && m_ready) begin
      if(m_tag!=seen || m_acc!==expected[seen] || m_overflow!==0 || m_second_valid!==1 || m_last!==(seen==31))
        $fatal(1,"frozen model mismatch case=%0d tag=%0d expected=%h got=%h",case_id,m_tag,expected[seen],m_acc);
      seen=seen+1;checked=checked+16;
    end
    if(cycles>2000000) $fatal(1,"timeout");
  end
  initial begin
    fd=$fopen("../model_vectors/cases.txt","r");
    if(!fd) $fatal(1,"missing frozen vectors; run model_vectors.py first");
    rc=$fscanf(fd,"%d",ncases);if(rc!=1) $fatal(1,"bad count");
    repeat(3) @(negedge clk);resetn=1;
    for(case_id=0;case_id<ncases;case_id++) begin
      @(negedge clk);
      rc=$fscanf(fd,"%d %d %d %d %h %h %d",iw,ih,groups,kernel,cfg_bias,multipliers,relui);
      if(rc!=7) $fatal(1,"bad header");
      relu=relui;seen=0;q_seen=0;
      for(int n=0;n<ih*((iw+1)/2)*2*groups;n++) begin
        @(negedge clk);
        rc=$fscanf(fd,"%d %d %h",bank,addr,f_data);
        if(rc!=3) $fatal(1,"bad feature");
        f_page=case_id%2;f_bank=bank;f_addr=addr;f_valid=1;
      end
      @(negedge clk);f_valid=0;
      for(int n=0;n<8*kernel*kernel*groups;n++) begin
        @(negedge clk);
        rc=$fscanf(fd,"%d %d %h",lane,addr,w_data);
        if(rc!=3) $fatal(1,"bad weight");
        w_lane=lane;w_addr=addr;w_valid=1;
      end
      @(negedge clk);w_valid=0;
      for(int n=0;n<32;n++) begin
        rc=$fscanf(fd,"%h %h",expected[n],expected_quant[n]);if(rc!=2) $fatal(1,"bad golden");
      end
      cfg_iw=iw;cfg_ih=ih;cfg_ci_groups=groups;cfg_k3=kernel==3;cfg_page=case_id%2;start=1;
      @(negedge clk);start=0;
      if(!busy || error) $fatal(1,"descriptor rejected");
      while(busy || q_seen<32) begin
        q_ready=$unsigned($random(seed))%5!=0;
        @(negedge clk);
      end
      if(seen!=32) $fatal(1,"incomplete fixture");
    end
    $fclose(fd);
    $display("PASS model_tiles checked=%0d blocks=%0d frozen_layers=30 quantized_equal=1",checked,ncases);$finish;
  end
endmodule
