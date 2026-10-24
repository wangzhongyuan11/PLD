`timescale 1ns/1ps
module tb_row_transform;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0,start=0,cfg_table=0;
  logic [1:0] cfg_mode=0;
  logic [9:0] cfg_width=0;
  logic busy,done,error,lut_table=0,lut_valid=0,lut_ready;
  logic [7:0] lut_index=0,lut_value=0;
  logic [63:0] s_data=0,m_data;
  logic s_valid=0,s_ready,m_valid,m_last,m_ready=0;
  row_transform dut(.*);
  logic [2047:0] lut_values;
  logic [63:0] inputs [0:1023],expected [0:511];
  integer fd,ncases,rc,groups,height,source_width,target_width,mode,rowmode,case_id,cg,y,words,seen=0,checked=0,seed=2814,cycles=0;
  logic held=0;
  logic [64:0] held_data;
  always @(posedge clk) begin
    cycles=cycles+1;
    if(resetn) begin
      if(held && (!m_valid || {m_last,m_data}!==held_data)) $fatal(1,"row transform unstable");
      held=m_valid && !m_ready;held_data={m_last,m_data};
      if(m_valid && m_ready) begin
        if(seen>=target_width || m_data!==expected[seen] || m_last!==(seen==target_width-1))
          $fatal(1,"full transition mismatch case=%0d cg=%0d y=%0d x=%0d expected=%h got=%h",case_id,cg,y,seen,expected[seen],m_data);
        seen=seen+1;checked=checked+1;
      end
    end
    if(cycles>12000000) $fatal(1,"timeout");
  end
  initial begin
    fd=$fopen("../model_vectors/transitions.txt","r");
    if(!fd) $fatal(1,"missing transition vectors");
    rc=$fscanf(fd,"%d",ncases);if(rc!=1) $fatal(1,"case count");
    repeat(4) @(negedge clk);resetn=1;
    for(case_id=0;case_id<ncases;case_id++) begin
      rc=$fscanf(fd,"%d %d %d %d %d %h",groups,height,source_width,target_width,mode,lut_values);
      if(rc!=6) $fatal(1,"header");
      for(int v=0;v<256;v++) begin @(negedge clk);lut_valid=1;lut_index=v;lut_value=lut_values[v*8+:8];end
      @(negedge clk);lut_valid=0;
      for(cg=0;cg<groups;cg++) for(y=0;y<height;y++) begin
        rc=$fscanf(fd,"%d",rowmode);if(rc!=1) $fatal(1,"row mode");
        words=rowmode==3 ? 0 : rowmode==1 ? source_width*2 : source_width;
        for(int n=0;n<words;n++) begin rc=$fscanf(fd,"%h",inputs[n]);if(rc!=1) $fatal(1,"source");end
        for(int n=0;n<target_width;n++) begin rc=$fscanf(fd,"%h",expected[n]);if(rc!=1) $fatal(1,"golden");end
        @(negedge clk);seen=0;cfg_mode=rowmode;cfg_width=source_width;start=1;
        @(negedge clk);start=0;
        for(int n=0;n<words;n++) begin
          @(negedge clk);s_valid=1;s_data=inputs[n];m_ready=$unsigned($random(seed))%5!=0;
          @(posedge clk);
          while(!s_ready) begin @(negedge clk);m_ready=$unsigned($random(seed))%5!=0;@(posedge clk);end
        end
        @(negedge clk);s_valid=0;
        while(busy) begin m_ready=$unsigned($random(seed))%5!=0;@(negedge clk);end
        if(error || seen!=target_width) $fatal(1,"row missing output");
      end
    end
    $fclose(fd);$display("PASS row_transform checked=%0d branches=%0d full_size=1",checked,ncases);$finish;
  end
endmodule
