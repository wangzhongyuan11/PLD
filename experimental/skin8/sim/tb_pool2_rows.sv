`timescale 1ns/1ps
module tb_pool2_rows;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0,start=0;
  logic [9:0] cfg_width=0;
  logic busy,done,error;
  logic [63:0] s_data=0,m_data;
  logic s_valid=0,s_ready,m_valid,m_last,m_ready=0;
  pool2_rows dut(.*);
  integer seed=222,seen=0,checked=0,epoch=0,value,largest;
  logic held=0;
  logic [64:0] held_data;
  function automatic integer pixel(input integer row,x,lane);
    return ((row*101+x*37+lane*29+epoch*13)%256)-128;
  endfunction
  always @(posedge clk) begin
    if(!resetn) held=0;
    else begin
      if(held && (!m_valid || {m_last,m_data}!==held_data)) $fatal(1,"pool output unstable");
      held=m_valid && !m_ready;held_data={m_last,m_data};
      if(m_valid && m_ready) begin
        if(m_last!==(seen==cfg_width/2-1)) $fatal(1,"pool last mismatch");
        for(int lane=0;lane<8;lane++) begin
          largest=-128;
          for(int y=0;y<2;y++) for(int x=0;x<2;x++) begin
            value=pixel(y,seen*2+x,lane);if(value>largest) largest=value;
          end
          if($signed(m_data[lane*8+:8])!=largest) $fatal(1,"signed pool mismatch");
          checked=checked+1;
        end
        seen=seen+1;
      end
    end
  end
  task automatic run_case(input integer width);
    @(negedge clk);cfg_width=width;start=1;seen=0;epoch=epoch+1;
    @(negedge clk);start=0;
    for(int n=0;n<width*2;n++) begin
      @(negedge clk);s_valid=1;m_ready=$unsigned($random(seed))%3!=0;
      for(int lane=0;lane<8;lane++) s_data[lane*8+:8]=pixel(n/width,n%width,lane);
      @(posedge clk);
      while(!s_ready) begin @(negedge clk);m_ready=$unsigned($random(seed))%3!=0;@(posedge clk);end
    end
    @(negedge clk);s_valid=0;m_ready=1;
    while(busy) @(negedge clk);
    if(error || seen!=width/2) $fatal(1,"pool missing output");
  endtask
  initial begin
    repeat(3) @(negedge clk);resetn=1;
    run_case(2);run_case(4);run_case(8);run_case(64);run_case(256);run_case(512);
    @(negedge clk);cfg_width=3;start=1;
    @(negedge clk);start=0;
    if(!error || busy) $fatal(1,"odd pool width accepted");
    $display("PASS pool2_rows checked=%0d",checked);$finish;
  end
endmodule
