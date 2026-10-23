`timescale 1ns/1ps
module tb_input_transform;
  logic aclk=0,aresetn=0,lut_we=0;
  logic [8:0] lut_addr=0;
  logic [7:0] lut_data=0,s_lane_mask=0;
  logic [63:0] s_data=0,m_data;
  logic [31:0] s_tag=0,m_tag;
  logic s_lookup=0,s_branch=0,s_zero=0,s_valid=0,s_ready,m_valid,m_ready=0,invalid_int8;
  logic [7:0] lut_values[512];
  logic [95:0] expected[2048],held;
  bit stalled=0,checking=1;
  int fd,ef,rc,n,received=0;
  int unsigned rng=32'hacdc2560;
  input_transform dut(.*);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  always @(negedge aclk)begin rng=next_rng(rng);m_ready=aresetn&&(rng[0]||rng[4]);end
  always @(posedge aclk)if(aresetn)begin
    if(stalled&&(!m_valid||{m_tag,m_data}!==held))$fatal(1,"transform changed under stall");
    stalled=m_valid&&!m_ready;held={m_tag,m_data};
    if(m_valid&&m_ready&&checking)begin
      if(received>=n||{m_tag,m_data}!==expected[received])$fatal(1,"transform mismatch %0d",received);
      received++;
    end
  end else stalled=0;
  initial begin
    $readmemh("lut.txt",lut_values);
    fd=$fopen("transform_input.txt","r");ef=$fopen("transform_expected.txt","r");
    if(!fd||!ef)$fatal(1,"missing transform vectors");
    rc=$fscanf(fd,"%d\n",n);if(rc!=1||n>2048)$fatal(1,"transform header");
    for(int j=0;j<n;j++)begin rc=$fscanf(ef,"%h\n",expected[j]);if(rc!=1)$fatal(1,"short expected");end
    repeat(4)@(negedge aclk);aresetn=1;
    for(int j=0;j<512;j++)begin @(negedge aclk);lut_we=1;lut_addr=j;lut_data=lut_values[j];end
    @(negedge aclk);lut_we=0;
    for(int j=0;j<n;j++)begin
      @(negedge aclk);s_valid=0;
      if(j%19==0)repeat(3)@(negedge aclk);
      rc=$fscanf(fd,"%h %h %h %h %h %h\n",s_lookup,s_branch,s_zero,s_lane_mask,s_tag,s_data);
      if(rc!=6)$fatal(1,"short transform input");s_valid=1;
      do @(posedge aclk);while(!s_ready);
    end
    @(negedge aclk);s_valid=0;wait(received==n);repeat(4)@(negedge aclk);
    if(invalid_int8)$fatal(1,"legal input flagged invalid");
    checking=0;s_valid=1;s_zero=0;s_data=64'h8080808080808080;s_lane_mask=8'hff;s_lookup=0;
    do @(posedge aclk);while(!s_ready);
    @(negedge aclk);s_valid=0;if(!invalid_int8)$fatal(1,"missing invalid -128 detection");
    aresetn=0;repeat(3)@(negedge aclk);aresetn=1;
    s_zero=1;s_valid=1;
    do @(posedge aclk);while(!s_ready);
    @(negedge aclk);s_valid=0;
    wait(m_valid);if(m_data!==64'd0||invalid_int8)$fatal(1,"padding not forced to zero");
    $display("TB_PASS input_transform words=%0d LUTs=512 raw_lookup_mask_padding_invalid_reset=PASS",received);$finish;
  end
  initial begin #1000000;$fatal(1,"transform watchdog");end
endmodule
