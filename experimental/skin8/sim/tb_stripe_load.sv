`timescale 1ns/1ps
module tb_stripe_load;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0,start=0;
  logic [31:0] cfg_source=32'h10000;
  logic [8:0] cfg_width=0,cfg_height=0,cfg_y=0,cfg_rows=0;
  logic [4:0] cfg_groups=0;
  logic cfg_pad=0,cfg_page=0,cfg_table=0,busy,done,error;
  logic lut_table=0,lut_valid=0,lut_ready;
  logic [7:0] lut_index=0,lut_value=0;
  logic [31:0] rd_addr;
  logic [15:0] rd_words;
  logic rd_cmd_valid,rd_cmd_ready,rd_valid,rd_last,rd_done,rd_error,rd_ready;
  logic [63:0] rd_data,f_data;
  logic f_page,f_bank,f_valid,f_ready=0;
  logic [11:0] f_addr;
  stripe_load dut(.*);
  logic [31:0] araddr;
  logic [7:0] arlen;
  logic [2:0] arsize;
  logic [1:0] arburst,rresp=0;
  logic arvalid,arready=0,rlast=0,rvalid=0,rready;
  logic [63:0] rdata=0;
  ddr_read64 reader(.clk,.resetn,.cmd_addr(rd_addr),.cmd_words(rd_words),.cmd_valid(rd_cmd_valid),.cmd_ready(rd_cmd_ready),
    .done(rd_done),.error(rd_error),.araddr,.arlen,.arsize,.arburst,.arvalid,.arready,.rdata,.rresp,.rlast,.rvalid,.rready,
    .m_data(rd_data),.m_valid(rd_valid),.m_last(rd_last),.m_ready(rd_ready));
  logic [63:0] banks [0:1][0:4095];
  integer pending=0,index=0,seed=4096,case_id=0,checked=0,cycles=0,baseword=0;
  logic consumed=0;
  function automatic logic [63:0] memory_word(input integer word_index);
    for(int lane=0;lane<8;lane++) memory_word[lane*8+:8]=((word_index*17+lane*23+case_id*7)%255)-127;
  endfunction
  function automatic logic [7:0] lut(input integer tableid,value);
    return tableid ? (value^8'h5a) : value;
  endfunction
  always @(posedge clk) begin
    cycles=cycles+1;consumed=rvalid && rready;
    if(resetn) begin
      if(arvalid && arready) begin
        if(pending || araddr<cfg_source || araddr+(arlen+1)*8>cfg_source+cfg_width*cfg_height*cfg_groups*8)
          $fatal(1,"out of bounds DDR read");
        pending=arlen+1;index=0;baseword=(araddr-cfg_source)/8;
      end
      if(consumed) begin pending=pending-1;index=index+1;end
      if(f_valid && f_ready) begin
        if(f_page!=cfg_page) $fatal(1,"wrong feature page");
        banks[f_bank][f_addr]=f_data;
      end
    end
    if(cycles>100000) $fatal(1,"timeout");
  end
  always @(negedge clk) begin
    arready=$unsigned($random(seed))%3!=0;f_ready=$unsigned($random(seed))%3!=0;
    if(!rvalid || consumed) begin
      rvalid=pending>0 && $unsigned($random(seed))%4!=0;
      rdata=memory_word(baseword+index);rlast=pending==1;
    end
  end
  task automatic run_case(input integer w,h,y,rows,groups,pad,tableid);
    integer iw,ih,sx,sy,addr;
    logic [63:0] expected;
    @(negedge clk);cfg_width=w;cfg_height=h;cfg_y=y;cfg_rows=rows;cfg_groups=groups;
    cfg_pad=pad;cfg_table=tableid;cfg_page=case_id%2;start=1;
    @(negedge clk);start=0;
    while(!done) @(negedge clk);
    if(error) $fatal(1,"valid stripe rejected");
    iw=w+pad*2;ih=rows+pad*2;
    for(int yy=0;yy<ih;yy++) for(int xx=0;xx<((iw+1)/2)*2;xx++) for(int cg=0;cg<groups;cg++) begin
      sx=xx-pad;sy=y+yy-pad;expected=0;
      if(xx<iw && sx>=0 && sx<w && sy>=0 && sy<h) begin
        expected=memory_word((cg*h+sy)*w+sx);
        for(int lane=0;lane<8;lane++) expected[lane*8+:8]=lut(tableid,expected[lane*8+:8]);
      end
      addr=(yy*((iw+1)/2)+xx/2)*groups+cg;
      if(banks[xx%2][addr]!==expected) $fatal(1,"stripe mismatch case=%0d x=%0d y=%0d cg=%0d",case_id,xx,yy,cg);
      checked=checked+1;
    end
    case_id=case_id+1;
  endtask
  initial begin
    repeat(4) @(negedge clk);resetn=1;
    for(int t=0;t<2;t++) for(int v=0;v<256;v++) begin
      @(negedge clk);lut_table=t;lut_index=v;lut_value=lut(t,v);lut_valid=1;
    end
    @(negedge clk);lut_valid=0;
    run_case(8,9,0,3,3,1,0);
    run_case(8,9,3,4,12,1,1);
    run_case(8,9,8,1,24,1,1);
    run_case(7,5,0,5,1,0,0);
    run_case(7,5,1,2,3,1,1);
    $display("PASS stripe_load checked=%0d cases=%0d",checked,case_id);$finish;
  end
endmodule
