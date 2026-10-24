`timescale 1ns/1ps
module tb_conv_pair_tile;
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
  logic m_last,m_second_valid,m_overflow,m_valid,m_ready=0;
  conv_pair_tile dut(.*);
  integer ci,iw,ih,ow,oh,kernel,groups,case_id=0,seen=0,total_checked=0,seed=192128,cycles=0;
  integer xx,yy,channel,expected;
  longint signed acc;
  logic held=0;
  logic [530:0] held_data;
  function automatic integer input_value(input integer c,y,x);
    if(c>=ci || y<0 || y>=ih || x<0 || x>=iw) return 0;
    if(kernel==3 && (y==0 || y==ih-1 || x==0 || x==iw-1)) return 0;
    return ((c*31+y*7+x*13+case_id*17)%255)-127;
  endfunction
  function automatic integer weight_value(input integer o,c,ky,kx);
    if(c>=ci) return 0;
    return ((o*19+c*13+ky*5+kx*11+case_id*23)%255)-127;
  endfunction
  always @(posedge clk) begin
    cycles=cycles+1;
    if(!resetn) held=0;
    else begin
      if(held && (!m_valid || {m_acc,m_tag,m_last,m_second_valid,m_overflow}!==held_data))
        $fatal(1,"stalled tile result changed");
      held=m_valid && !m_ready;held_data={m_acc,m_tag,m_last,m_second_valid,m_overflow};
      if(m_valid && m_ready) begin
        if(m_tag!=seen || m_last!==(seen==oh*((ow+1)/2)-1)) $fatal(1,"tag/order failure");
        yy=seen/((ow+1)/2);xx=(seen%((ow+1)/2))*2;
        if(m_second_valid!==(xx+1<ow) || m_overflow!==0) $fatal(1,"flags failure");
        for(int p=0;p<2;p++)
          for(int o=0;o<8;o++) if(xx+p<ow) begin
            acc=$signed(cfg_bias[o*32+:32]);
            for(int c=0;c<ci;c++)
              for(int ky=0;ky<kernel;ky++)
                for(int kx=0;kx<kernel;kx++)
                  acc=acc+input_value(c,yy+ky,xx+p+kx)*weight_value(o,c,ky,kx);
            expected=acc;
            if(m_acc[(p*8+o)*32+:32]!==expected)
              $fatal(1,"tile mismatch case=%0d tag=%0d p=%0d o=%0d expect=%0d got=%0d",case_id,m_tag,p,o,expected,$signed(m_acc[(p*8+o)*32+:32]));
            total_checked=total_checked+1;
          end
        seen=seen+1;
      end
    end
    if(cycles>500000) $fatal(1,"timeout");
  end
  task automatic run_case(input integer channels,width,height,k);
    ci=channels;iw=width;ih=height;kernel=k;ow=iw-k+1;oh=ih-k+1;groups=(ci+7)/8;seen=0;
    for(int y=0;y<ih;y++)
      for(int x=0;x<((iw+1)/2)*2;x++)
        for(int g=0;g<groups;g++) begin
          @(negedge clk);f_valid=1;f_page=case_id%2;f_bank=x%2;f_addr=(y*((iw+1)/2)+x/2)*groups+g;
          for(int c=0;c<8;c++) f_data[c*8+:8]=input_value(g*8+c,y,x);
        end
    @(negedge clk);f_valid=0;
    for(int o=0;o<8;o++)
      for(int ky=0;ky<kernel;ky++)
        for(int kx=0;kx<kernel;kx++)
          for(int g=0;g<groups;g++) begin
            @(negedge clk);w_valid=1;w_lane=o;w_addr=(ky*kernel+kx)*groups+g;
            for(int c=0;c<8;c++) w_data[c*8+:8]=weight_value(o,g*8+c,ky,kx);
          end
    @(negedge clk);w_valid=0;
    cfg_page=case_id%2;cfg_k3=kernel==3;cfg_iw=iw;cfg_ih=ih;cfg_ci_groups=groups;
    for(int o=0;o<8;o++) cfg_bias[o*32+:32]=((o+case_id)%2 ? -1 : 1)*(o+1)*123;
    start=1;
    @(negedge clk);start=0;
    if(!busy || error) $fatal(1,"legal descriptor rejected");
    while(busy) begin
      // Load inactive page while the active page supplies the MAC continuously.
      f_valid=1;f_page=!cfg_page;f_bank=$random(seed);f_addr=$random(seed);f_data={$random(seed),$random(seed)};
      m_ready=$unsigned($random(seed))%5!=0;
      @(negedge clk);
    end
    f_valid=0;m_ready=1;
    if(seen!=oh*((ow+1)/2)) $fatal(1,"missing tile results");
    $display("case %0d ci=%0d iw=%0d ih=%0d k=%0d passed",case_id,ci,iw,ih,kernel);
    case_id=case_id+1;
  endtask
  initial begin
    repeat(3) @(negedge clk);resetn=1;
    run_case(3,9,7,3);
    run_case(8,10,8,3);
    run_case(16,7,8,3);
    run_case(24,8,7,3);
    run_case(32,5,5,3);
    run_case(48,6,5,3);
    run_case(64,5,6,3);
    run_case(96,5,5,3);
    run_case(128,5,5,3);
    run_case(192,5,5,3);
    run_case(8,9,7,1);
    run_case(16,8,6,1);
    @(negedge clk);cfg_ci_groups=0;start=1;
    @(negedge clk);start=0;
    if(!error || busy) $fatal(1,"zero channels accepted");
    @(negedge clk);cfg_ci_groups=24;cfg_iw=258;cfg_ih=258;start=1;
    @(negedge clk);start=0;
    while(busy) @(negedge clk);
    if(!error || busy) $fatal(1,"oversized descriptor accepted");
    $display("PASS conv_pair_tile checked=%0d cases=%0d",total_checked,case_id);$finish;
  end
endmodule
