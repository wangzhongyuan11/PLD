`timescale 1ns/1ps
module tb_graph_control;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0,start=0,busy,done,error;
  logic [31:0] cfg_descriptors=32'h1000,rd_addr;
  logic [15:0] cfg_count=3,completed,rd_words;
  logic rd_cmd_valid,rd_cmd_ready,rd_valid=0,rd_last=0,rd_error=0,rd_ready;
  logic [63:0] rd_data=0;
  logic engine_start,engine_busy=0,engine_done=0,engine_error=0;
  logic [1023:0] descriptor;
  logic lut_valid,lut_table,lut_ready=0;
  logic [7:0] lut_index,lut_value;
  graph_control dut(.*);
  integer pending=0,index=0,seed=119,cycles=0,engine_delay=0,launches=0,lut_bytes=0,checked=0;
  integer inject=0,descid=0;
  logic [31:0] address;
  logic consumed=0;
  assign rd_cmd_ready=pending==0 && !rd_valid;
  function automatic logic [63:0] word(input integer a,n);
    integer id;
    logic [1023:0] d;
    if(a<32'h4000) begin
      id=(a-32'h1000)/128;d=0;d[224+:32]=id;
      d[223]=id!=1;d[256+:32]=32'h4000;
      if(inject==3) d[288]=1;
      return d[n*64+:64];
    end
    for(int b=0;b<8;b++) word[b*8+:8]=n*8+b;
  endfunction
  always @(posedge clk) begin
    cycles=cycles+1;consumed=rd_valid && rd_ready;engine_done<=0;
    if(resetn) begin
      if(rd_cmd_valid && rd_cmd_ready) begin
        pending=rd_words;index=0;address=rd_addr;rd_error=0;
      end
      if(consumed) begin pending=pending-1;index=index+1;end
      if(lut_valid && lut_ready) begin
        if(lut_value!==lut_index || lut_table!==(lut_bytes%512>=256)) $fatal(1,"wrong LUT mapping");
        lut_bytes=lut_bytes+1;checked=checked+1;
      end
      if(engine_start) begin
        if(engine_busy || pending || rd_valid) $fatal(1,"engine launched before reader drained");
        if(descriptor[224+:32]!==launches) $fatal(1,"wrong descriptor order");
        if(lut_bytes!=(launches==2 ? 1024 : 512)) $fatal(1,"incomplete LUT before launch");
        launches=launches+1;engine_busy<=1;engine_delay=7;
      end else if(engine_delay>0) begin
        engine_delay=engine_delay-1;
        if(engine_delay==0) begin engine_busy<=0;engine_done<=1;engine_error<=inject==4 && launches==2;end
      end
    end
    if(cycles>30000) $fatal(1,"controller timeout state=%0d pending=%0d launches=%0d bytes=%0d",dut.state,pending,launches,lut_bytes);
  end
  always @(negedge clk) begin
    lut_ready=$unsigned($random(seed))%3!=0;
    if(!rd_valid || consumed) begin
      rd_valid=pending>0 && $unsigned($random(seed))%3!=0;
      rd_data=word(address,index);rd_last=pending==1;
      if(rd_valid && ((inject==1 && address<32'h4000) || (inject==2 && address>=32'h4000))) rd_error=1;
    end
  end
  task automatic run(input integer fault,expected_count,expected_launches);
    @(negedge clk);inject=fault;launches=0;lut_bytes=0;engine_error=0;start=1;
    @(negedge clk);start=0;while(!done) @(negedge clk);
    if(error!==(fault!=0) || completed!=expected_count || launches!=expected_launches || pending!=0)
      $fatal(1,"wrong recovery status fault=%0d completed=%0d launches=%0d",fault,completed,launches);
    checked=checked+1;
  endtask
  initial begin
    repeat(4) @(negedge clk);resetn=1;
    run(0,3,3);run(1,0,0);run(0,3,3);run(2,0,0);run(0,3,3);
    run(3,0,0);run(4,1,2);run(0,3,3);
    @(negedge clk);cfg_count=0;start=1;
    @(negedge clk);start=0;if(!done || !error) $fatal(1,"empty list accepted");
    $display("PASS graph_control checked=%0d descriptor_lut_engine_faults=1",checked);$finish;
  end
endmodule
