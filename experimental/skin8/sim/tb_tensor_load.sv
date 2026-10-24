`timescale 1ns/1ps
module tb_tensor_load;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0,start=0;
  logic [31:0] cfg_source=32'h10000;
  logic [8:0] cfg_width=0,cfg_height=0,cfg_y=0,cfg_rows=0;
  logic [4:0] cfg_groups=0;
  logic [31:0] cfg_epoch=1;
  logic [4:0] cfg_branch_groups=0,cfg_group_offset=0;
  logic [1:0] cfg_mode=0;
  logic cfg_clear=1;
  integer source_w,source_h;
  logic cfg_pad=0,cfg_page=0,cfg_table=0,busy,done,error;
  logic lut_table=0,lut_valid=0,lut_ready;
  logic [7:0] lut_index=0,lut_value=0;
  logic [31:0] rd_addr;
  logic [15:0] rd_words;
  logic rd_cmd_valid,rd_cmd_ready,rd_valid,rd_last,rd_done,rd_error,rd_ready;
  logic [63:0] rd_data,f_data;
  logic f_page,f_bank,f_valid,f_ready=0;
  logic [11:0] f_addr;
  tensor_load dut(.*);
  logic [31:0] araddr;
  logic [7:0] arlen;
  logic [2:0] arsize;
  logic [1:0] arburst,rresp=0;
  logic arvalid,arready=0,rlast=0,rvalid=0,rready;
  logic [63:0] rdata=0;
  ddr_read64 reader(.clk,.resetn,.cmd_addr(rd_addr),.cmd_words(rd_words),.cmd_valid(rd_cmd_valid),.cmd_ready(rd_cmd_ready),
    .done(rd_done),.error(rd_error),.araddr,.arlen,.arsize,.arburst,.arvalid,.arready,.rdata,.rresp,.rlast,.rvalid,.rready,
    .m_data(rd_data),.m_valid(rd_valid),.m_last(rd_last),.m_ready(rd_ready));
  logic [63:0] saved [0:1][0:4095];
  logic [63:0] banks [0:1][0:4095];
  integer pending=0,index=0,seed=4096,case_id=0,checked=0,cycles=0,baseword=0;
  logic consumed=0;
  integer commands=0,command_start=0;
  logic inject_error=0;
  function automatic logic [63:0] memory_word(input integer word_index);
    for(int lane=0;lane<8;lane++) memory_word[lane*8+:8]=((word_index*17+lane*23+case_id*7)%255)-127;
  endfunction
  function automatic logic [7:0] lut(input integer tableid,value);
    return tableid ? (value^8'h5a) : value;
  endfunction
  always @(posedge clk) begin
    cycles=cycles+1;consumed=rvalid && rready;
    if(resetn) begin
      if(rd_cmd_valid && rd_cmd_ready) commands=commands+1;
      if(arvalid && arready) begin
        if(pending || araddr<cfg_source || araddr+(arlen+1)*8>cfg_source+source_w*source_h*cfg_branch_groups*8)
          $fatal(1,"out of bounds DDR read");
        pending=arlen+1;index=0;baseword=(araddr-cfg_source)/8;
      end
      if(consumed) begin pending=pending-1;index=index+1;end
      if(f_valid && f_ready) begin
        if(f_page!=cfg_page) $fatal(1,"wrong feature page");
        banks[f_bank][f_addr]=f_data;
      end
    end
    if(cycles>2000000) $fatal(1,"timeout");
  end
  always @(negedge clk) begin
    arready=$unsigned($random(seed))%3!=0;f_ready=$unsigned($random(seed))%3!=0;
    if(!rvalid || consumed) begin
      rvalid=pending>0 && $unsigned($random(seed))%4!=0;
      rdata=memory_word(baseword+index);rlast=pending==1;rresp=inject_error ? 2'b10 : 0;
    end
  end
  task automatic run_case(input integer w,h,y,rows,groups,pad,tableid,mode,offset,clearpage);
    integer iw,ih,sx,sy,addr;
    logic [63:0] expected,tmp;
    logic signed [7:0] maximum,value;
    @(negedge clk);cfg_width=w;cfg_height=h;cfg_y=y;cfg_rows=rows;cfg_groups=groups;
    cfg_pad=pad;cfg_table=tableid;if(clearpage) cfg_page=case_id%2;
    cfg_mode=mode;cfg_branch_groups=groups-offset;cfg_group_offset=offset;cfg_clear=clearpage;
    source_w=mode==1 ? 2*w : mode==2 ? w/2 : w;
    source_h=mode==1 ? 2*h : mode==2 ? h/2 : h;command_start=commands;start=1;
    @(negedge clk);start=0;
    while(!done) @(negedge clk);
    if(error) $fatal(1,"valid stripe rejected");
    begin
      integer firstrow,lastrow,expected_commands;
      firstrow=(pad && y>0) ? y-1 : y;
      lastrow=(y+rows+pad>h) ? h : y+rows+pad;
      expected_commands=(groups-offset)*(mode==2 ? ((lastrow-1)/2-firstrow/2+1) : lastrow-firstrow);
      if(commands-command_start!=expected_commands) $fatal(1,"unexpected row rereads mode=%0d got=%0d expected=%0d",mode,commands-command_start,expected_commands);
    end
    iw=w+pad*2;ih=rows+pad*2;
    for(int yy=0;yy<ih;yy++) for(int xx=0;xx<((iw+1)/2)*2;xx++) for(int cg=offset;cg<groups;cg++) begin
      sx=xx-pad;sy=y+yy-pad;expected=0;
      if(xx<iw && sx>=0 && sx<w && sy>=0 && sy<h) begin
        if(mode==0) expected=memory_word(((cg-offset)*source_h+sy)*source_w+sx);
        else if(mode==2) expected=memory_word(((cg-offset)*source_h+sy/2)*source_w+sx/2);
        else begin
          for(int lane=0;lane<8;lane++) begin
            maximum=-128;
            for(int dy=0;dy<2;dy++) for(int dx=0;dx<2;dx++) begin
              tmp=memory_word(((cg-offset)*source_h+sy*2+dy)*source_w+sx*2+dx);
              value=$signed(tmp[lane*8+:8]);if(value>maximum) maximum=value;
            end
            expected[lane*8+:8]=maximum;
          end
        end
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
    run_case(8,9,0,3,3,1,0,0,0,1);
    run_case(8,9,3,4,12,1,1,1,0,1);
    run_case(8,10,8,2,24,1,1,2,0,1);
    run_case(7,5,0,5,1,0,0,0,0,1);
    run_case(7,5,1,2,3,1,1,1,0,1);
    run_case(32,32,3,7,8,1,0,2,0,1);
    run_case(256,256,0,8,2,1,1,1,0,1);
    run_case(256,256,247,9,2,1,1,2,0,1);
    // Append a normal branch without clearing earlier groups.
    run_case(32,32,3,7,8,1,0,2,0,1);
    for(int b=0;b<2;b++) for(int a=0;a<4096;a++) saved[b][a]=banks[b][a];
    run_case(32,32,3,7,8,1,1,0,4,0);
    for(int b=0;b<2;b++) for(int a=0;a<17*9*8;a++)
      if(a%8<4 && saved[b][a]!==banks[b][a]) $fatal(1,"append overwrote first branch");
    @(negedge clk);cfg_epoch=2;start=1;
    @(negedge clk);start=0;while(!done) @(negedge clk);
    if(!error) $fatal(1,"stale append accepted");
    // AXI failure drains the outstanding response, then allows a fresh reload.
    @(negedge clk);cfg_clear=1;inject_error=1;start=1;
    @(negedge clk);start=0;while(!done) @(negedge clk);
    if(!error || pending!=0) $fatal(1,"failed read did not drain");
    @(negedge clk);inject_error=0;
    run_case(32,32,3,7,8,1,0,2,0,1);
    // LUT modification must invalidate append eligibility.
    @(negedge clk);lut_valid=1;lut_table=0;lut_index=0;lut_value=0;
    @(negedge clk);lut_valid=0;cfg_clear=0;start=1;
    @(negedge clk);start=0;while(!done) @(negedge clk);
    if(!error) $fatal(1,"append accepted after LUT change");
    // Reject unaligned, oversized-bank and invalid branch descriptors before DDR.
    for(int invalid_case=0;invalid_case<3;invalid_case++) begin
      @(negedge clk);cfg_clear=1;cfg_source=32'h10000;cfg_groups=8;cfg_branch_groups=8;
      cfg_group_offset=0;cfg_width=32;cfg_height=32;cfg_y=0;cfg_rows=7;cfg_mode=0;
      if(invalid_case==0) cfg_source=32'h10001;
      if(invalid_case==1) begin cfg_width=256;cfg_height=256;cfg_rows=256;end
      if(invalid_case==2) cfg_group_offset=1;
      command_start=commands;start=1;
      @(negedge clk);start=0;while(!done) @(negedge clk);
      if(!error || commands!=command_start) $fatal(1,"invalid descriptor reached DDR");
    end
    $display("PASS tensor_load checked=%0d cases=%0d",checked,case_id);$finish;
  end
endmodule
